import { dummyTodo } from "../types/todos";
import TodoItem from "./TodoItem";


interface TodoListProps {
    todos: dummyTodo[];
    onCompletedChange: (id: number, completed: boolean) => void;
    onDelete: (id: number) => void;
}

export default function TodoList({
    todos,
    onCompletedChange,
    onDelete
}: TodoListProps) {
    
    const todosSorted = todos.sort((a, b) => {
        if (a.completed === b.completed) {
            return b.id - a.id;
        }
        return a.completed ? 1 : -1;
    }
    );
    
    return (
        <>
        <div>
          {todosSorted.map((todo) => <TodoItem 
                                      key={todo.id}
                                      todo={todo}
                                      onCompletedChange={onCompletedChange}
                                      onDelete={onDelete}/> )}
        </div>
        {todos.length === 0 && (
            <div className="bg-amber-100 p-4 m-2 rounded-lg">
                <p className="text-center text-gray-500">No todos yet!</p>
            </div>
        )}
        </>
    )
    }