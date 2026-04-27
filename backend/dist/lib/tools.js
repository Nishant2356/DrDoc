import { ChatGroq } from "@langchain/groq";
export const simplellm = new ChatGroq({
    model: "meta-llama/llama-4-scout-17b-16e-instruct", // <-- Use a valid Groq model
    apiKey: process.env.GROQ_API_KEY,
    temperature: 0,
    maxRetries: 2,
});
