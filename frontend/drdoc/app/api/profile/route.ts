import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/app/lib/db";

export async function PUT(req: NextRequest) {
  try {
    const { userId, specialties, commonMedicines, notePreferences } =
      await req.json();

    // Validation
    if (!userId) {
      return NextResponse.json(
        { error: "User ID is required." },
        { status: 400 }
      );
    }

    if (!specialties?.length) {
      return NextResponse.json(
        { error: "At least one specialty is required." },
        { status: 400 }
      );
    }

    if (!commonMedicines?.length) {
      return NextResponse.json(
        { error: "At least one medicine is required." },
        { status: 400 }
      );
    }

    if (!notePreferences?.trim()) {
      return NextResponse.json(
        { error: "Note preferences are required." },
        { status: 400 }
      );
    }

    // Upsert clinical profile (create if doesn't exist, update if it does)
    const profile = await prisma.clinicalProfile.upsert({
      where: { userId },
      update: {
        specialties,
        commonMedicines,
        notePreferences: notePreferences.trim(),
      },
      create: {
        userId,
        specialties,
        commonMedicines,
        notePreferences: notePreferences.trim(),
      },
    });

    return NextResponse.json({
      clinicalProfile: {
        specialties: profile.specialties,
        commonMedicines: profile.commonMedicines,
        notePreferences: profile.notePreferences,
      },
    });
  } catch (error) {
    console.error("Profile update error:", error);
    return NextResponse.json(
      { error: "Internal server error." },
      { status: 500 }
    );
  }
}
