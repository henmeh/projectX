import AddTodoForm from "./components/AddTodoForm";
import TodoList from "./components/TodoList";
import TodoSummary from "./components/TodoSummary";
import useTodos from "./hooks/useTodos";

function App() {

  const {
    todos,
    setToDoCompleted,
    deleteTodo,
    deleteAllCompleted,
    addTodo
  } = useTodos();

  return (
    <main className="py-10 h-screen overflow-auto">
      <h1 className="font-bold text-3xl text-center text-amber-500">My ToDo's</h1>
      <div className="max-w-lg mx-auto">
      <AddTodoForm onSubmit={addTodo}/>
      <TodoList todos={todos} onCompletedChange={setToDoCompleted} onDelete={deleteTodo}/>
      <TodoSummary todos={todos} deleteAllCompleted={deleteAllCompleted}/>
      </div>
    </main>
  )
}

export default App
