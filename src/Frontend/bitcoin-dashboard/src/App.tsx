import { useState } from "react"
import { dummyTodos } from "./data/todos"
import AddTodoForm from "./components/AddTodoForm";
import TodoList from "./components/TodoList";

function App() {

  const [todos, setTodos] = useState(dummyTodos);

  function setToDoCompleted(id: number, completed: boolean) {
    setTodos((prevTodos) => prevTodos.map((todo) => {
      if (todo.id === id) {
        return { ...todo, completed }
      }
      return todo;
    }
    ))
  }

  function deleteTodo(id: number) {
    setTodos((prevTodos) => prevTodos.filter(todo => todo.id !== id))
  }

  function addTodo(title: string) {
    setTodos((prevTodos) => [
      {
        id: prevTodos.length + 1,
        title,
        completed: false,
      },     
      ...prevTodos]);
  }

  return (
    <main className="py-10 h-screen overflow-auto">
      <h1 className="font-bold text-3xl text-center text-amber-500">My ToDo's</h1>
      <div className="max-w-lg mx-auto">
      <AddTodoForm onSubmit={addTodo}/>
      <TodoList todos={todos} onCompletedChange={setToDoCompleted} onDelete={deleteTodo}/>
      </div>
    </main>
  )
}

export default App
