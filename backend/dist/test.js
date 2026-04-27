import drDocAgent from "./index.ts";
const result = drDocAgent.invoke({
    originalTranscript: "hello i am your first prototype agent"
});
console.log("result", result);
