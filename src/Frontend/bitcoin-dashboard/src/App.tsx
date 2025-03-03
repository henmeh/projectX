import { dummyTodos } from "./data/todos"
import TodoItem from "./components/TodoItem"

function App() {

  
  return (
    <main className="py-10 h-screen">
      <h1 className="font-bold text-3xl text-center text-amber-500">My ToDo's</h1>
      <div className="max-w-lg mx-auto">
        <div>
          {dummyTodos.map((todo) => <TodoItem todo={todo}/> )}
        </div>
      </div>
    </main>
  )
}

export default App
