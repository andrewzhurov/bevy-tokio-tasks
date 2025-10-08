use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use bevy_app::{App, Plugin, Update};
use bevy_ecs::schedule::{InternedScheduleLabel, ScheduleLabel};
use bevy_ecs::{prelude::World, resource::Resource};

use tokio::{runtime::Runtime, task::JoinHandle};

/// A re-export of the tokio version used by this crate.
pub use tokio;

/// An internal struct keeping track of how many ticks have elapsed since the start of the program.
#[derive(Resource)]
struct UpdateTicks {
    current_tick: Arc<AtomicUsize>,
    update_watch_tx: tokio::sync::watch::Sender<()>,
}

impl UpdateTicks {
    fn tick(&self) -> usize {
        let new_tick = self
            .current_tick
            .fetch_add(1, Ordering::SeqCst)
            .wrapping_add(1);
        self.update_watch_tx
            .send(())
            .expect("Failed to send update_watch channel message");
        new_tick
    }
}

/// The Bevy [`Plugin`] which sets up the [`TokioTasksRuntime`] Bevy resource and registers
/// the [`tick_runtime_update`] exclusive system.
pub struct TokioTasksPlugin {
    /// Callback which is used to create a Tokio runtime when the plugin is installed. The
    /// default value for this field configures a multi-threaded [`Runtime`] with IO and timer
    /// functionality enabled if building for non-wasm32 architectures. On wasm32 the current-thread
    /// scheduler is used instead.
    pub make_runtime: Box<dyn Fn() -> Runtime + Send + Sync + 'static>,
    /// The [`ScheduleLabel`] during which the [`tick_runtime_update`] function will be executed.
    /// The default value for this field is [`Update`].
    pub schedule_label: InternedScheduleLabel,
}

impl Default for TokioTasksPlugin {
    /// Configures the plugin to build a new Tokio [`Runtime`] with both IO and timer functionality
    /// enabled. On the wasm32 architecture, the [`Runtime`] will be the current-thread runtime, on all other
    /// architectures the [`Runtime`] will be the multi-thread runtime.
    ///
    /// The default schedule label is [`Update`].
    fn default() -> Self {
        Self {
            make_runtime: Box::new(|| {
                #[cfg(not(target_arch = "wasm32"))]
                let mut runtime = tokio::runtime::Builder::new_multi_thread();
                #[cfg(target_arch = "wasm32")]
                let mut runtime = tokio::runtime::Builder::new_current_thread();
                runtime.enable_all();
                runtime
                    .build()
                    .expect("Failed to create Tokio runtime for background tasks")
            }),
            schedule_label: Update.intern(),
        }
    }
}

impl Plugin for TokioTasksPlugin {
    fn build(&self, app: &mut App) {
        let current_tick = Arc::new(AtomicUsize::new(0));
        let (update_watch_tx, update_watch_rx) = tokio::sync::watch::channel(());
        let runtime = (self.make_runtime)();
        app.insert_resource(UpdateTicks {
            current_tick: current_tick.clone(),
            update_watch_tx,
        });
        app.insert_resource(TokioTasksRuntime::new(
            current_tick,
            runtime,
            update_watch_rx,
        ));
        app.add_systems(self.schedule_label, tick_runtime_update);
    }
}

/// The Bevy exclusive system which executes the main thread callbacks that background
/// tasks requested using [`run_on_main_thread`](TaskContext::run_on_main_thread). You
/// can control which Bevy schedule stage this system executes in by specifying a custom
/// [`schedule_label`](TokioTasksPlugin::schedule_label) value.
pub fn tick_runtime_update(world: &mut World) {
    let current_tick = {
        let Some(tick_counter) = world.get_resource::<UpdateTicks>() else {
            return;
        };

        // Increment update ticks and notify watchers of update tick.
        tick_counter.tick()
    };
    execute_main_thread_work(world, current_tick);
}

type MainThreadCallback = Box<dyn FnOnce(MainThreadContext) + Send + 'static>;

/// The Bevy [`Resource`] which stores the Tokio [`Runtime`] and allows for spawning new
/// background tasks.
#[derive(Resource)]
pub struct TokioTasksRuntime(Box<TokioTasksRuntimeInner>);

/// The inner fields are boxed to reduce the cost of the every-frame move out of and back into
/// the world in [`tick_runtime_update`].
struct TokioTasksRuntimeInner {
    runtime: Runtime,
    current_tick: Arc<AtomicUsize>,
    update_watch_rx: tokio::sync::watch::Receiver<()>,
    main_cb_tx: tokio::sync::mpsc::UnboundedSender<MainThreadCallback>,
    main_cb_rx: tokio::sync::mpsc::UnboundedReceiver<MainThreadCallback>,
}

impl TokioTasksRuntime {
    fn new(
        current_tick: Arc<AtomicUsize>,
        runtime: Runtime,
        update_watch_rx: tokio::sync::watch::Receiver<()>,
    ) -> Self {
        let (main_cb_tx, main_cb_rx) = tokio::sync::mpsc::unbounded_channel();

        Self(Box::new(TokioTasksRuntimeInner {
            runtime,
            current_tick,
            update_watch_rx,
            main_cb_tx,
            main_cb_rx,
        }))
    }

    /// Returns the Tokio [`Runtime`] on which background tasks are executed. You can specify
    /// how this is created by providing a custom [`make_runtime`](TokioTasksPlugin::make_runtime).
    pub fn runtime(&self) -> &Runtime {
        &self.0.runtime
    }

    /// Spawn a task which will run on the background Tokio [`Runtime`] managed by this [`TokioTasksRuntime`]. The
    /// background task is provided a [`TaskContext`] which allows it to do things like
    /// [sleep for a given number of main thread updates](TaskContext::sleep_updates) or
    /// [invoke callbacks on the main Bevy thread](TaskContext::run_on_main_thread).
    pub fn spawn_background_task<Task, Output, Spawnable>(
        &self,
        spawnable_task: Spawnable,
    ) -> JoinHandle<Output>
    where
        Task: Future<Output = Output> + Send + 'static,
        Output: Send + 'static,
        Spawnable: FnOnce(TaskContext) -> Task + Send + 'static,
    {
        let inner = &self.0;
        let context = TaskContext {
            update_watch_rx: inner.update_watch_rx.clone(),
            main_cb_tx: inner.main_cb_tx.clone(),
            current_tick: inner.current_tick.clone(),
        };
        let future = spawnable_task(context);
        inner.runtime.spawn(future)
    }
}

// A function, rather than a method, to not require taking `TokioTasksRuntime` out of World.
// As World callbocks which it executes may expect it to be present.
/// Execute all of the requested runnables on the main thread.
pub(crate) fn execute_main_thread_work(world: &mut World, current_tick: usize) {
    // Running this single future which yields once allows the runtime to process tasks
    // if the runtime is a current_thread runtime. If its a multi-thread runtime then
    // this isn't necessary but is harmless.
    if let Some(tt_runtime) = world.get_resource::<TokioTasksRuntime>() {
        tt_runtime.runtime().block_on(async {
            tokio::task::yield_now().await;
        });
    }

    while let Some(runnable) = world
        .get_resource_mut::<TokioTasksRuntime>()
        .and_then(|mut tt_runtime| tt_runtime.0.main_cb_rx.try_recv().ok())
    {
        let context = MainThreadContext {
            world,
            current_tick,
        };
        runnable(context);
    }
}

/// The context arguments which are available to main thread callbacks requested using
/// [`run_on_main_thread`](TaskContext::run_on_main_thread).
pub struct MainThreadContext<'a> {
    /// A mutable reference to the main Bevy [World].
    pub world: &'a mut World,
    /// The current update tick in which the current main thread callback is executing.
    pub current_tick: usize,
}

/// The context arguments which are available to background tasks spawned onto the
/// [`TokioTasksRuntime`].
#[derive(Clone)]
pub struct TaskContext {
    update_watch_rx: tokio::sync::watch::Receiver<()>,
    main_cb_tx: tokio::sync::mpsc::UnboundedSender<MainThreadCallback>,
    current_tick: Arc<AtomicUsize>,
}

impl TaskContext {
    /// Returns the current value of the tick counter from the main thread - how many updates
    /// have occurred since the start of the program. Because the tick counter is updated from the
    /// main thread, it may change any time after this function call returns.
    pub fn current_tick(&self) -> usize {
        self.current_tick.load(Ordering::SeqCst)
    }

    /// Sleeps the background task until a given number of main thread updates have occurred.
    /// If you instead want to sleep for a given length of wall-clock time, call the normal Tokio sleep function.
    pub async fn sleep_updates(&mut self, updates_to_sleep: usize) {
        let target_tick = self.current_tick().wrapping_add(updates_to_sleep);
        while self.current_tick() < target_tick && self.update_watch_rx.changed().await.is_err() {}
    }

    /// Invokes a synchronous callback on the main Bevy thread. The callback will have mutable access to the
    /// main Bevy [`World`], allowing it to update any resources or entities that it wants. The callback can
    /// report results back to the background thread by returning an output value, which will then be returned from
    /// this async function once the callback runs.
    pub async fn run_on_main_thread<Runnable, Output>(&mut self, runnable: Runnable) -> Output
    where
        Runnable: FnOnce(MainThreadContext) -> Output + Send + 'static,
        Output: Send + 'static,
    {
        let (output_tx, output_rx) = tokio::sync::oneshot::channel();
        if self.main_cb_tx.send(Box::new(move |ctx| {
            if output_tx.send(runnable(ctx)).is_err() {
                panic!("Failed to sent output from operation run on main thread back to waiting task");
            }
         })).is_err() {
            panic!("Failed to send operation to be run on main thread");
        }
        output_rx
            .await
            .expect("Failed to receive output from operation on main thread")
    }
}
