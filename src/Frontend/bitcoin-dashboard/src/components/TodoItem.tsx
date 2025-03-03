import { dummyTodo } from "../types/todos";

interface TodoItemProps {
    todo: dummyTodo;
    onCompletedChange: (id: number, completed: boolean) => void;
    onDelete: (id: number) => void;
}

export default function TodoItem({todo, onCompletedChange, onDelete}: TodoItemProps) {
  return (
    <div className="bg-amber-100 p-4 m-2 rounded-lg">
        <label className="flex items-center gap-2">
            <input type="checkbox"
            checked={todo.completed}
            onChange={(e) => onCompletedChange(todo.id, e.target.checked)}
            />
            <span className={todo.completed ? "line-through text-gray-400" : ""}>{todo.title}</span>
        </label>
        <button onClick={()=>onDelete(todo.id)} className="bg-red-500">Delete</button>
    </div>
  )
}