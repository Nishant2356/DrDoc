import { StateGraph, START, END, MemorySaver } from "@langchain/langgraph";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { SystemMessage, HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { ChatGroq } from "@langchain/groq";
import { simplellm } from "./lib/tools.js";
import { autoCorrectorPrompt, feedbackIntegratorPrompt, generateNotePrompt, mockDoctorProfile, piiRedactionPrompt, rootCauseAnalyzerPrompt, safetyGuardrailPrompt, updateDatabasePrompt } from "./lib/prompts.js";

// ==========================================
// 1. STATE DEFINITION
// The "Memory" shared across all nodes
// ==========================================
export interface DrDocState {
    originalTranscript: string;
    redactedTranscript: string;
    doctorPreferences: string;
    draftNote: string;
    guardrailStatus: "PENDING" | "PASS" | "FAIL";
    guardrailError: string;
    humanReviewStatus: "PENDING" | "APPROVED" | "REJECTED";
    humanFeedbackText: string;
    rootCause: "TRANSCRIPT_ERROR" | "PREFERENCE_ERROR" | null;
}

// State channels (how LangGraph merges data between steps)
// Using a simple override for each field.
const graphState = {
    originalTranscript: { value: (x: any, y: any) => y ?? x, default: () => "" },
    redactedTranscript: { value: (x: any, y: any) => y ?? x, default: () => "" },
    doctorPreferences: { value: (x: any, y: any) => y ?? x, default: () => "" },
    draftNote: { value: (x: any, y: any) => y ?? x, default: () => "" },
    guardrailStatus: { value: (x: any, y: any) => y ?? x, default: () => "PENDING" },
    guardrailError: { value: (x: any, y: any) => y ?? x, default: () => "" },
    humanReviewStatus: { value: (x: any, y: any) => y ?? x, default: () => "PENDING" },
    humanFeedbackText: { value: (x: any, y: any) => y ?? x, default: () => "" },
    rootCause: { value: (x: any, y: any) => y ?? x, default: () => null },
};

// ==========================================
// 2. NODE FUNCTIONS (The AI Actions)
// ==========================================

const piiRedactionNode = async (state: DrDocState) => {
    console.log("🟢 [Node] Redacting PII with Gemini...");

    // 1. Initialize Gemini (gemini-1.5-flash is perfect for fast, high-volume text tasks)
    const llm = simplellm

    // 2. The System Prompt (Same strict rules)
    const systemPrompt = piiRedactionPrompt

    // 3. The User Prompt
    const userPrompt = new HumanMessage(state.originalTranscript);

    try {
        // 4. Call Gemini
        const response = await llm.invoke([systemPrompt, userPrompt]);
        //console.log(response.content)

        // 5. Return the updated state
        return { redactedTranscript: response.content as string };

    } catch (error) {
        console.error("❌ Gemini PII Redaction Failed:", error);
        // Fallback safety
        return { redactedTranscript: "ERROR: Transcription redaction failed. Please review manually." };
    }
};

const knowledgeRetrievalNode = async (state: DrDocState) => {
    console.log("🟢 [Node] Fetching RAG Doctor Preferences (Mock DB)...");

    // Simulate a 500ms database network call
    await new Promise((resolve) => setTimeout(resolve, 500));

    // The Mock Vector DB Output
    // This simulates what your database would return after searching the doctor's profile

    // Inject this retrieved knowledge into the graph's state
    return { doctorPreferences: mockDoctorProfile };
};

const generateNoteNode = async (state: DrDocState) => {
    console.log("🟢 [Node] Generating Clinical Note with Groq...");

    // 1. Initialize the smarter Groq model for complex reasoning
    const llm = simplellm

    // 2. Dynamically handle the Continuous Learning Loop
    // If the graph was rejected and looped back, we force the LLM to read the doctor's feedback.
    const systemPrompt = generateNotePrompt(state);

    // 4. The User Prompt (The input data)
    const userPrompt = new HumanMessage(`
      Raw Consultation Transcript:
      <transcript>
      ${state.redactedTranscript}
      </transcript>
    `);

    try {
        // 5. Generate the note
        const response = await llm.invoke([systemPrompt, userPrompt]);

        console.log(response.content);
        // 6. Return the updated draft to the state
        return { draftNote: response.content as string };

    } catch (error) {
        console.error("❌ Note Generation Failed:", error);
        return { draftNote: "ERROR: Clinical note generation failed due to API timeout." };
    }
};

const safetyGuardrailNode = async (state: DrDocState) => {
    console.log("🟢 [Node] Running Safety Guardrail...");
  
    // 1. Initialize Groq. We use the 70B model with zero temperature for strict logical evaluation.
    const llm = simplellm
  
    // 2. The System Prompt (The JSON Output Enforcer)
    const systemPrompt = safetyGuardrailPrompt;
  
    // 3. The User Prompt (Passing the draft note from the previous node)
    const userPrompt = new HumanMessage(`
      Draft Note to Evaluate:
      <draft>
      ${state.draftNote}
      </draft>
    `);
  
    try {
      // 4. Call the LLM to evaluate the draft
      const response = await llm.invoke([systemPrompt, userPrompt]);
      
      // 5. Clean and parse the JSON output
      // (We strip out markdown backticks just in case the LLM disobeys the 'no formatting' rule)
      const resultText = response.content as string;
      const cleanJsonText = resultText.replace(/```json/g, '').replace(/```/g, '').trim();
      
      const evaluation = JSON.parse(cleanJsonText);
  
      // 6. Route based on the LLM's true/false decision
      if (evaluation.passed) {
        console.log("   ✅ Guardrail Passed.");
        return { guardrailStatus: "PASS", guardrailError: "" };
      } else {
        console.log(`   ❌ Guardrail Failed: ${evaluation.error}`);
        return { guardrailStatus: "FAIL", guardrailError: evaluation.error };
      }
      
    } catch (error) {
      console.error("❌ Guardrail API or Parsing Failed:", error);
      // HACKATHON FLEX: The Ultimate Fail-Safe
      // If the Groq API times out or the JSON parsing fails, we DO NOT let the note pass.
      // We intentionally fail it so it routes to human review.
      return { 
        guardrailStatus: "FAIL", 
        guardrailError: "System error during automated safety check. Manual review required to verify dosages." 
      };
    }
};

const autoCorrectorNode = async (state: DrDocState) => {
    console.log("🟠 [Node] Auto-Correcting Safety Failure...");
  
    // 1. Initialize Groq (Zero temperature because we want strict, surgical fixes)
    const llm = simplellm
  
    // 2. The System Prompt (The Surgical Editor)
    const systemPrompt = autoCorrectorPrompt(state);
  
    // 3. The User Prompt (The Broken Draft + The Error)
    const userPrompt = new HumanMessage(`
      Failed Draft Note:
      <draft>
      ${state.draftNote}
      </draft>
  
      Safety Guardrail Error to Fix:
      <error>
      ${state.guardrailError}
      </error>
    `);
  
    try {
      // 4. Call Groq to perform the fix
      const response = await llm.invoke([systemPrompt, userPrompt]);
      
      console.log("   ✅ Auto-Correction Applied.");
      
      // 5. Return the fixed note (This will trigger the graph to loop back to the guardrail!)
      return { draftNote: response.content as string };
      
    } catch (error) {
      console.error("❌ Auto-Correction API Failed:", error);
      // HACKATHON FAIL-SAFE:
      // If the auto-corrector crashes, return the original broken draft.
      // The graph will catch it on the next loop or eventually pass it to the human review 
      // rather than destroying the draft completely.
      return { draftNote: state.draftNote }; 
    }
};

const humanHandoffNode = async (state: DrDocState) => {
    console.log("⏸️ [Node] Waiting for Physician Review...");
    
    // This node intentionally does absolutely nothing.
    // The LangGraph checkpointer freezes the entire application state right BEFORE this executes.
    return {}; 
};;

const feedbackIntegratorNode = async (state: DrDocState) => {
    console.log("🟣 [Node] Integrating Human Feedback...");
  
    // 1. Initialize Groq (Using the 70B model for high reasoning)
    const llm = simplellm
  
    // 2. The System Prompt (The Clinical Translator)
    const systemPrompt = feedbackIntegratorPrompt;
  
    // 3. The User Prompt (The Broken Draft + The Doctor's Angry Text)
    const userPrompt = new HumanMessage(`
      Previously Rejected Draft Note:
      <draft>
      ${state.draftNote}
      </draft>
  
      Physician's Raw Feedback:
      <feedback>
      ${state.humanFeedbackText}
      </feedback>
    `);
  
    try {
      // 4. Generate the structured instructions
      const response = await llm.invoke([systemPrompt, userPrompt]);
      const structuredInstructions = response.content as string;
      
      console.log("   ✅ Feedback translated into strict instructions.");
  
      // 5. Update the state
      // We overwrite the raw human feedback with our new, hyper-clear instructions.
      // We also clear out the draftNote so the Generator starts fresh!
      return { 
        humanFeedbackText: structuredInstructions,
        draftNote: "" // Wipe the slate clean for the next node
      };
      
    } catch (error) {
      console.error("❌ Feedback Integration Failed:", error);
      // Fallback: If the API fails, just pass the raw feedback through to the generator
      // so the loop doesn't break.
      return { 
        draftNote: "", 
        humanFeedbackText: `CRITICAL MANDATE: ${state.humanFeedbackText}` 
      };
    }
};

const rootCauseAnalyzerNode = async (state: DrDocState) => {
    console.log("🟣 [Node] Analyzing Root Cause in Background...");
  
    // 1. Initialize Groq (70B model because this requires deep logical deduction)
    const llm = simplellm
  
    // 2. The System Prompt (The Detective)
    const systemPrompt = rootCauseAnalyzerPrompt;
  
    // 3. The User Prompt (Providing all the evidence)
    const userPrompt = new HumanMessage(`
      Evidence File:
      
      1. Saved Database Rules: 
      <rules>${state.doctorPreferences}</rules>
      
      2. Original Transcript:
      <transcript>${state.originalTranscript}</transcript>
      
      3. The Rejected Draft Note:
      <draft>${state.draftNote}</draft>
      
      4. The Doctor's Rejection Feedback:
      <feedback>${state.humanFeedbackText}</feedback>
    `);
  
    try {
      // 4. Run the analysis
      const response = await llm.invoke([systemPrompt, userPrompt]);
      
      // 5. Parse the JSON
      const cleanJsonText = (response.content as string).replace(/```json/g, '').replace(/```/g, '').trim();
      const analysis = JSON.parse(cleanJsonText);
      
      console.log(`   🕵️ Root Cause Detected: ${analysis.rootCause}`);
      console.log(`   📝 Reasoning: ${analysis.reasoning}`);
  
      // 6. Return the classification to trigger the router
      return { rootCause: analysis.rootCause };
      
    } catch (error) {
      console.error("❌ Root Cause Analysis Failed:", error);
      // HACKATHON FAIL-SAFE:
      // If the detective fails to make a decision, we default to null. 
      // This safely kills the background job and prevents us from accidentally ruining the database.
      return { rootCause: null }; 
    }
};

const updateDatabaseNode = async (state: DrDocState) => {
    console.log("🟣 [Node] Updating Supabase Vector DB (Continuous Learning)...");
  
    // 1. Initialize Groq (70B model because rewriting protocols requires high accuracy)
    const llm = simplellm
  
    // 2. The System Prompt (The Database Administrator)
    const systemPrompt = updateDatabasePrompt;
  
    // 3. The User Prompt (Old Rules + The Change)
    // Note: Because this runs on a parallel branch, it has access to the exact feedback
    // that triggered the update.
    const userPrompt = new HumanMessage(`
      Current Saved Protocols:
      <old_protocols>
      ${state.doctorPreferences}
      </old_protocols>
  
      The Doctor's Permanent Change:
      <feedback>
      ${state.humanFeedbackText}
      </feedback>
    `);
  
    try {
      // 4. Generate the newly updated database string
      const response = await llm.invoke([systemPrompt, userPrompt]);
      const updatedDatabaseString = response.content as string;
      
      // 5. The Hackathon Demo Flex (Simulate the DB Save)
      console.log("\n   💾 --- SUPABASE UPDATE EXECUTED ---");
      console.log("   [OLD PROTOCOL DELETED]");
      console.log("   [NEW PROTOCOL SAVED]:\n   " + updatedDatabaseString.substring(0, 150) + "...\n");
  
      // 6. Return the updated preferences into the graph's state memory
      // So if the loop runs again, it uses the NEW rules!
      return { doctorPreferences: updatedDatabaseString };
      
    } catch (error) {
      console.error("❌ Database Update Failed:", error);
      // If it fails, we safely return nothing so the graph just ends cleanly
      return {}; 
    }
};

// ==========================================
// 3. ROUTING LOGIC (Conditional Edges)
// ==========================================

const checkSafetyStatus = (state: DrDocState) => {
    return state.guardrailStatus === "FAIL" ? "auto_correct" : "human_handoff";
};

const checkHumanDecision = (state: DrDocState) => {
    if (state.humanReviewStatus === "APPROVED") {
        return END;
    }
    // Return the array of node names directly to trigger parallel execution!
    return ["feedback_integrator", "root_cause_analyzer"];
};


const checkRootCause = (state: DrDocState) => {
    return state.rootCause === "PREFERENCE_ERROR" ? "update_db" : "end_background_job";
};

// ==========================================
// 4. COMPILE THE GRAPH
// ==========================================

const workflow = new StateGraph<DrDocState>({ channels: graphState as any })
    // Add Nodes
    .addNode("redact_pii", piiRedactionNode)
    .addNode("retrieve_knowledge", knowledgeRetrievalNode)
    .addNode("generate_note", generateNoteNode)
    .addNode("safety_guardrail", safetyGuardrailNode)
    .addNode("auto_correct", autoCorrectorNode)
    .addNode("human_handoff", humanHandoffNode)
    .addNode("feedback_integrator", feedbackIntegratorNode)
    .addNode("root_cause_analyzer", rootCauseAnalyzerNode)
    .addNode("update_database", updateDatabaseNode)

    // Linear Path
    .addEdge(START, "redact_pii")
    .addEdge("redact_pii", "retrieve_knowledge")
    .addEdge("retrieve_knowledge", "generate_note")
    .addEdge("generate_note", "safety_guardrail")

    // LOOP 1: Automated Safety QA
    .addConditionalEdges("safety_guardrail", checkSafetyStatus, {
        auto_correct: "auto_correct",
        human_handoff: "human_handoff",
    })
    .addEdge("auto_correct", "safety_guardrail")

    // LOOP 2: Human Review & Parallel Learning
    .addConditionalEdges(
        "human_handoff",
        checkHumanDecision,
        // Provide a flat array of ALL possible destination nodes for the static validator
        [END, "feedback_integrator", "root_cause_analyzer"]
    )


    // The Critical Path (Fix Note)
    .addEdge("feedback_integrator", "generate_note")

    // The Background Job (Fix Database)
    .addConditionalEdges("root_cause_analyzer", checkRootCause, {
        update_db: "update_database",
        end_background_job: END
    })
    .addEdge("update_database", END);

// 1. Initialize an in-memory checkpointer
const memory = new MemorySaver();

// 2. Add it to the compiled graph
export const drDocAgent = workflow.compile({
    checkpointer: memory, // <-- This saves the state!
    interruptBefore: ["human_handoff"],
});



async function runDrDocTest() {
    console.log("🚀 STARTING DR. DOC AGENT TEST...\n");

    // LangGraph uses thread IDs to remember the state when it pauses for the Human Handoff.
    const config = { configurable: { thread_id: "test_consultation_001" } };

    // ==========================================
    // PHASE 1: The Initial Consultation
    // ==========================================
    //console.log("▶️ PHASE 1: Simulating live consultation...");

    const initialInput = {
        originalTranscript: "Hi Dr. Smith, thank you for fitting me in today. For the front desk records, my name is Michael Scott, my phone number is 555-867-5309, and my date of birth is March 15th, 1980. Anyway, I've been feeling absolutely terrible. I've had a severe, pounding headache right behind my eyes for about four days now, and my sinuses feel completely completely blocked. I took my temperature this morning and it was 101.2. I haven't taken anything for it except some over-the-counter Tylenol, but it isn't touching the pain. Also, just as a reminder for my chart, I am severely allergic to penicillin—it gives me full-body hives. Doctor speaking : Alright Michael, let's take a look. Yes, there is significant inflammation and purulent discharge in the nasal cavity. This is definitely an acute sinus infection. Given your symptoms, we need to hit this with an antibiotic. I know my standard protocol is usually Amoxicillin for this, but since you have that penicillin allergy, I'm going to prescribe Azithromycin instead. I'm also going to prescribe some Fluticasone nasal spray to help bring down that swelling. Make sure you get plenty of rest, drink fluids, and let's have you follow up in 10 days if things don't improve."
    };

    // We invoke the agent. It will run through PII Redaction -> RAG -> Generation -> Guardrail
    // and then PAUSE exactly at the "human_handoff" node as we configured.
    const firstRunState = await drDocAgent.invoke(initialInput, config);

    // console.log("\n⏸️ GRAPH PAUSED AT HUMAN HANDOFF.");
    // console.log("Current Draft Note:", firstRunState.draftNote);
    // console.log("--------------------------------------------------\n");

    // ==========================================
    // PHASE 2: Simulating Doctor's Rejection
    // ==========================================
    console.log("▶️ PHASE 2: Doctor clicks 'Reject' and gives feedback...");

    // 1. Explicitly update the state for this thread
    await drDocAgent.updateState(config, {
        humanReviewStatus: "REJECTED",
        humanFeedbackText: "Wait, Amoxicillin doesn't treat migraines. Change it to Sumatriptan 50mg.",
    });

    // 2. Invoke with 'null' to tell it to just resume from where it paused
    const secondRunState = await drDocAgent.invoke(null, config);

    // console.log("\n⏸️ GRAPH PAUSED AT HUMAN HANDOFF (Iteration 2).");
    // console.log("Updated Draft Note:", secondRunState.draftNote);
    // console.log("--------------------------------------------------\n");

    // ==========================================
    // PHASE 3: Simulating Doctor's Approval
    // ==========================================
    console.log("▶️ PHASE 3: Doctor reviews the new draft and clicks 'Approve'...");

    // 1. Update the state to approved
    await drDocAgent.updateState(config, {
        humanReviewStatus: "APPROVED",
    });

    // 2. Resume the graph to finish
    const finalState = await drDocAgent.invoke(null, config);

    console.log(finalState)

    // console.log("\n✅ GRAPH FINISHED.");
    // console.log("Final Approved Note:", finalState.draftNote);
    // console.log("Ready to push to EHR and Print!");
}

// Execute the test
runDrDocTest().catch(console.error);

// ==========================================
// USAGE EXAMPLE (How to call this in your API route)
// ==========================================
/*
  // 1. Initial Run
  const thread = { configurable: { thread_id: "consultation_123" } };
  await drDocAgent.invoke({ originalTranscript: "Raw text from Deepgram..." }, thread);
  
  // 2. Doctor Responds (Resume the graph)
  await drDocAgent.invoke({ 
    humanReviewStatus: "REJECTED", 
    humanFeedbackText: "Change amoxicillin to 1000mg" 
  }, thread);
*/
