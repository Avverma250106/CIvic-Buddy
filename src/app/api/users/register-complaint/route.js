import { connectDB } from "@/utils/connectDB";
import { NextResponse } from "next/server";
import complaintModel from "@/models/complaint.model";
import { uploadToCloudinary } from "@/utils/cloudinary-upload";
import { getDataFromToken } from "@/utils/getDataFromToken";
import User from "@/models/user.model";

connectDB();

// FastAPI endpoint configuration
const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL || "http://127.0.0.1:8000";

/**
 * Call FastAPI to classify the image and get department assignment
 */
async function classifyComplaintImage(imageUrl) {
  try {
    const response = await fetch(`${FASTAPI_BASE_URL}/predict-url`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        image_url: imageUrl,
      }),
    });

    if (!response.ok) {
      throw new Error(`FastAPI returned status ${response.status}`);
    }

    const data = await response.json();
    
    return {
      success: true,
      category: data.prediction.class,
      department: data.prediction.department,
      confidence: data.prediction.confidence,
      description: data.prediction.description,
      usedGemini: data.prediction.used_gemini,
      allProbabilities: data.all_probabilities,
    };
  } catch (error) {
    console.error("Error calling FastAPI:", error);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * Map department names from FastAPI to your department slugs
 */
function mapDepartmentToSlug(department) {
  const departmentMap = {
    'PWD': 'pwd',
    'Electricity': 'electricity',
    'Water': 'water',
    'NRMC': 'nrmc',
    'Environment': 'environment',
  };
  
  return departmentMap[department] || 'nrda';
}

export async function POST(req) {
  try {
    // Deconstructing everything from form
    const formData = await req.formData();
    const file = formData.get("file");
    const issueType = formData.get("issue-type");
    const description = formData.get("description");
    const latitude = formData.get("latitude");
    const longitude = formData.get("longitude");
    const address = formData.get("address");
    const manualDepartment = formData.get("assigned-dept"); // Optional manual override

    const userId = getDataFromToken(req);

    if (!userId) {
      return NextResponse.json({
        success: false,
        message: "Unauthorized - Invalid token",
        statusCode: 401,
      });
    }

    const user = await User.findById(userId);

    if (!user) {
      return NextResponse.json({
        success: false,
        message: "Invalid user",
        statusCode: 404,
      });
    }

    if (!file) {
      return NextResponse.json({
        success: false,
        message: "File not found",
        statusCode: 400,
      });
    }

    if (!description) {
      return NextResponse.json({
        success: false,
        message: "Description is required",
        statusCode: 400,
      });
    }

    // ===== STEP 1: Upload file to Cloudinary =====
    console.log("📤 Uploading image to Cloudinary...");
    const imageUrl = await uploadToCloudinary(file, "civic-buddy");
    console.log("✅ Image uploaded:", imageUrl);

    // ===== STEP 2: Classify image using FastAPI =====
    console.log("🤖 Classifying image with AI model...");
    const classification = await classifyComplaintImage(imageUrl);
    
    let assignedDepartment;
    let aiCategory;
    let aiConfidence;
    let aiDescription;
    
    if (classification.success) {
      console.log("✅ Classification successful:", classification);
      
      // Convert department name to slug
      assignedDepartment = mapDepartmentToSlug(classification.department);
      aiCategory = classification.category;
      aiConfidence = classification.confidence;
      aiDescription = classification.description;
      
      // Allow manual override if provided
      if (manualDepartment && manualDepartment.trim() !== '') {
        console.log("⚠️ Manual department override:", manualDepartment);
        assignedDepartment = manualDepartment;
      }
    } else {
      console.error("❌ Classification failed:", classification.error);
      
      // Fallback: use manual department or default
      assignedDepartment = manualDepartment || 'general';
      aiCategory = issueType || 'other';
      aiConfidence = 0;
      aiDescription = "Classification unavailable";
    }

    // ===== STEP 3: Prepare location data =====
    const locationData = {};
    
    if (latitude && longitude) {
      locationData.latitude = parseFloat(latitude);
      locationData.longitude = parseFloat(longitude);
    }
    
    if (address) {
      locationData.address = address.trim();
    }

    // ===== STEP 4: Create and save complaint =====
    const newComplaint = new complaintModel({
      createdBy: userId,
      imageUrl: imageUrl,
      issueType: issueType || aiCategory || 'other',
      description: description.trim(),
      assignedDepartment: assignedDepartment,
      location: locationData,
      
      // Store AI classification metadata
      aiClassification: {
        category: aiCategory,
        confidence: aiConfidence,
        department: classification.success ? classification.department : null,
        description: aiDescription,
        classifiedAt: new Date(),
        usedGemini: classification.usedGemini || false,
      },
    });

    await newComplaint.save();

    console.log("✅ Complaint saved successfully:", newComplaint._id);

    // ===== STEP 5: Return response =====
    return NextResponse.json({
      success: true,
      message: "Complaint registered successfully",
      complaint: {
        id: newComplaint._id,
        imageUrl: newComplaint.imageUrl,
        issueType: newComplaint.issueType,
        assignedDepartment: newComplaint.assignedDepartment,
        location: newComplaint.location,
        createdAt: newComplaint.createdAt,
      },
      classification: classification.success ? {
        category: aiCategory,
        department: classification.department,
        departmentSlug: assignedDepartment,
        confidence: aiConfidence,
        description: aiDescription,
      } : null,
      statusCode: 201,
    });

  } catch (error) {
    console.error("❌ Error occurred during complaint registration:", error);
    return NextResponse.json({
      success: false,
      message: error.message || "Complaint registration failed",
      statusCode: 500,
    });
  }
}