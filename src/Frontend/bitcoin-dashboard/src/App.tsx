import { useState } from "react"
import { dummyTodos } from "./data/todos"
import TodoItem from "./components/TodoItem"
import AddTodoForm from "./components/AddTodoForm";

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
    <main className="py-10 h-screen">
      <h1 className="font-bold text-3xl text-center text-amber-500">My ToDo's</h1>
      <div className="max-w-lg mx-auto">
      <AddTodoForm onSubmit={addTodo}/>
        <div>
          {todos.map((todo) => <TodoItem 
                                      key={todo.id}
                                      todo={todo}
                                      onCompletedChange={setToDoCompleted}/> )}
        </div>
      </div>
    </main>
  )
}

export default App
