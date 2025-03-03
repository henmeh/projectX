import { useState } from "react"

interface AddTodoFormProps {
    onSubmit: (title: string) => void;
}

export default function AddTodoForm({onSubmit}: AddTodoFormProps) {

    const [inputValue, setInputValue] = useState("");

    function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
        e.preventDefault();
        if (inputValue.trim() === "") return;

        onSubmit(inputValue);
        setInputValue(""); // Clear the input after submission
    }

    return(
        <form className="flex" onSubmit={handleSubmit}>
            <input
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Enter new Todo"
                className="rounded-s-md grow border border-amber-500 p-2"
            />
            <button type="submit" className="w-16 rounded-e-md bg-slate-900 text-white hover:bg-slate-600">Add</button>
        </form>
    )
}