import { NextRequest, NextResponse } from "next/server";
import { drDocAgent } from "../../lib/agent/index";
import { prisma } from "../../lib/db";

export async function POST(req: NextRequest) {
  try {
    const { threadId, noteId, feedback } = await req.json();

    if (!threadId || !noteId || !feedback) {
      return NextResponse.json({ error: "Missing required fields" }, { status: 400 });
    }

    const thread = { configurable: { thread_id: threadId } };
    
    console.log(`Processing feedback for thread ${threadId}: ${feedback}`);

    // Update the state at the human_handoff node to trigger rejection path
    await drDocAgent.updateState(
      thread,
      {
        humanReviewStatus: "REJECTED",
        humanFeedbackText: feedback,
      },
      "human_handoff"
    );

    // Resume the agent so it processes the feedback and generates a new note
    const newState = await drDocAgent.invoke(null, thread);
    
    console.log("Agent finished corrected draft:", newState.draftNote);

    // Update the note in the database
    await prisma.generatedNote.update({
      where: { id: noteId },
      data: {
        noteContent: newState.draftNote,
      },
    });
    
    // Also try to update the continuous learning profile if the graph changed it
    // We would need the user's email to do this cleanly, but since we are modifying
    // their preferences we can get it from the note relation
    const note = await prisma.generatedNote.findUnique({
      where: { id: noteId },
      include: { user: true }
    });

    if (note && note.user) {
      // If the agent changed the preferences based on the root cause analysis, save them
      if (newState.doctorPreferences && newState.doctorPreferences !== "") {
        await prisma.clinicalProfile.update({
          where: { userId: note.user.id },
          data: { notePreferences: newState.doctorPreferences }
        });
        console.log("Updated doctor profile with continuous learning feedback.");
      }
    }

    return NextResponse.json({ 
      note: newState.draftNote, 
    });

  } catch (err) {
    console.error("Note correction error:", err);
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
