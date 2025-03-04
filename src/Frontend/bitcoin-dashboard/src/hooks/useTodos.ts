import { useEffect, useState } from "react";
import { dummyTodo } from "../types/todos";
import { dummyTodos } from "../data/todos";

export default function useTodos() {
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

      return {
        todos,
        setToDoCompleted,
        deleteTodo,
        deleteAllCompleted,
        addTodo
      }
}