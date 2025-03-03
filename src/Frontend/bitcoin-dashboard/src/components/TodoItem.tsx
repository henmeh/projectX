import { dummyTodo } from "../types/todos";

interface TodoItemProps {
    todo: dummyTodo;
}

export default function TodoItem({todo}: TodoItemProps) {
  return (
    <div className="bg-amber-100 p-4 m-2 rounded-lg">
      <h2 className="text-xl">{todo.title}</h2>
      <p>{todo.completed ? "Completed" : "Not Completed"}</p>
    </div>
  )
}