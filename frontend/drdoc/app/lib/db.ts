import { PrismaClient } from "../../prisma/generated/client";
// Import the driver adapter for your specific database (example uses PostgreSQL)
import { PrismaPg } from "@prisma/adapter-pg";
// Initialize the adapter according to your driver's requirements
const adapter = new PrismaPg({ connectionString: process.env.DATABASE_URL });
// Pass the adapter instance to PrismaClient
export const prisma = new PrismaClient({ adapter });