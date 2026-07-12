import { PrismaClient } from "./prisma/generated/client";
import { Pool } from "pg";
import { PrismaPg } from "@prisma/adapter-pg";
import bcrypt from 'bcryptjs';

const connectionString = "postgresql://neondb_owner:npg_u2xXnGWc4SoF@ep-rapid-mountain-aoparei2-pooler.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require";

const pool = new Pool({ connectionString });
const adapter = new PrismaPg(pool);
const prisma = new PrismaClient({ adapter });

async function seed() {
  const email = "nm2356opp@gmail.com";
  let user = await prisma.user.findUnique({ where: { email } });

  if (!user) {
    const hashedPassword = await bcrypt.hash("password123", 10);
    user = await prisma.user.create({
      data: {
        email,
        name: "Dr. NM",
        password: hashedPassword,
        role: "DOCTOR"
      }
    });
    console.log("Created user", user.email);
  } else {
    console.log("Found existing user", user.email);
  }

  const profile = await prisma.clinicalProfile.upsert({
    where: { userId: user.id },
    update: {
      specialties: ["General Practice", "Internal Medicine"],
      commonMedicines: ["Amoxicillin", "Lisinopril", "Metformin", "Atorvastatin", "Ibuprofen"],
      notePreferences: "Strictly use the SOAP format (Subjective, Objective, Assessment, Plan).\nKeep sentences professional, concise, and objective.\nDo not use abbreviations unless universally recognized.\nAlways document follow-up plans clearly."
    },
    create: {
      userId: user.id,
      specialties: ["General Practice", "Internal Medicine"],
      commonMedicines: ["Amoxicillin", "Lisinopril", "Metformin", "Atorvastatin", "Ibuprofen"],
      notePreferences: "Strictly use the SOAP format (Subjective, Objective, Assessment, Plan).\nKeep sentences professional, concise, and objective.\nDo not use abbreviations unless universally recognized.\nAlways document follow-up plans clearly."
    }
  });

  console.log("Upserted Clinical Profile for", user.email);
  console.log(profile);
}

seed().catch(console.error).finally(() => prisma.$disconnect());
