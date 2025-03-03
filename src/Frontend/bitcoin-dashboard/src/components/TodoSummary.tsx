import { dummyTodo } from "../types/todos";

interface TodoSummaryProps {
    todos: dummyTodo[];
    deleteAllCompleted: () => void;
}

export default function TodoSummary({todos, deleteAllCompleted}: TodoSummaryProps) {
    
    const completedTodos = todos.filter(todo => todo.completed);    
    return (
        <div>
            <h3>Completed Todos: {completedTodos.length}</h3>
            {
                completedTodos.length > 0 && <button onClick={()=>deleteAllCompleted()} className="bg-red-500">Delete all completed</button>
            }
        </div>
    );
}