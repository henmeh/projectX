import { useEffect, useState } from "react"
import { dummyTodos } from "./data/todos"
import AddTodoForm from "./components/AddTodoForm";
import TodoList from "./components/TodoList";
import TodoSummary from "./components/TodoSummary";
import { dummyTodo } from "./types/todos";

function App() {

  const [todos, setTodos] = useState(() => {
    const storedTodos: dummyTodo[] = JSON.parse(localStorage.getItem("todos") || "[]");
    return storedTodos.length > 0 ? storedTodos : dummyTodos;
  });

  useEffect(() => {
    localStorage.setItem("todos", JSON.stringify(todos));
  }, [todos])

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

  function deleteAllCompleted() {
    setTodos((prevTodos) => prevTodos.filter(todo => !todo.completed))
  }

  function addTodo(title: string) {
    setTodos((prevTodos) => [
      {
        id: Date.now(),
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
      <TodoSummary todos={todos} deleteAllCompleted={deleteAllCompleted}/>
      </div>
    </main>
  )
}

export default App
