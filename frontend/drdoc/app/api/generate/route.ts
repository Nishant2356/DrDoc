import { NextRequest, NextResponse } from "next/server";
import { drDocAgent } from "../../lib/agent/index";
import { prisma } from "../../lib/db";
import Groq from "groq-sdk";

// Initialize Groq SDK for transcription
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const audioBlob = formData.get("audio") as Blob;
    const email = formData.get("email") as string;

    if (!audioBlob) {
      return NextResponse.json({ error: "No audio provided" }, { status: 400 });
    }

    // 1. Convert the Blob to a File object for the Groq SDK
    const file = new File([audioBlob], "recording.webm", { type: "audio/webm" });

    // 2. Transcribe Audio using Whisper API via Groq
    console.log("Transcribing audio...");
    const transcription = await groq.audio.transcriptions.create({
      file: file,
      model: "whisper-large-v3",
      response_format: "verbose_json",
    });

    const transcriptText = transcription.text;
    console.log("Transcription successful:", transcriptText);

    // 3. Initialize the LangGraph agent for Note Generation
    console.log("Running Dr. Doc LangGraph Agent...");
    const thread = { configurable: { thread_id: `consultation_${Date.now()}` } };
    
    const initialState = { 
        originalTranscript: transcriptText 
    };
    
    // Invoke runs the graph until it finishes or hits an interrupt
    const state = await drDocAgent.invoke(initialState, thread);

    console.log("Agent finished draft:", state.draftNote);

    // 4. Save to Database
    let noteId = null;
    if (email) {
      const user = await prisma.user.findUnique({ where: { email } });
      if (user) {
        const savedNote = await prisma.generatedNote.create({
          data: {
            userId: user.id,
            patientInfo: state.patientInfo || {},
            noteContent: state.draftNote as string,
            originalTranscript: transcriptText
          }
        });
        noteId = savedNote.id;
        console.log("Note saved to database successfully.");
      }
    }

    return NextResponse.json({ 
      note: state.draftNote, 
      transcript: transcriptText,
      threadId: thread.configurable.thread_id,
      noteId: noteId
    });

  } catch (err) {
    console.error("Note generation error:", err);
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
