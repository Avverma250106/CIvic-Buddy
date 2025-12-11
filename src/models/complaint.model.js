import mongoose from "mongoose";

const complaintSchema = new mongoose.Schema(
  {
    createdBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    imageUrl: {
      type: String,
      required: true,
    },
    issueType: {
      type: String,
      enum: [
        "pothole",
        "garbage_dumping",
        "water_supply",
        "sewage",
        "road_damage",
        "traffic_signal",
        "noise_pollution",
        "streetlight_broken",
        "water_logging",
        "illegal_construction",
        "other",
      ],
      default: 'other',
    },
    description: {
      type: String,
      required: true,
      trim: true,
    },
    assignedDepartment: {
      type: String,
      enum: ['pwd', 'electricity', 'water', 'nrmc', 'environment', 'nrda'],
      required: true,
    },
    status: {
      type: String,
      enum: ['pending', 'in_progress', 'resolved', 'rejected'],
      default: 'pending',
    },
    location: {
      latitude: {
        type: Number,
        min: -90,
        max: 90,
      },
      longitude: {
        type: Number,
        min: -180,
        max: 180,
      },
      address: {
        type: String,
        trim: true,
      },
    },
    // AI Classification metadata
    aiClassification: {
      category: {
        type: String,
        default: null,
      },
      confidence: {
        type: Number,
        min: 0,
        max: 100,
        default: 0,
      },
      department: {
        type: String,
        default: null,
      },
      description: {
        type: String,
        default: null,
      },
      classifiedAt: {
        type: Date,
        default: null,
      },
      usedGemini: {
        type: Boolean,
        default: false,
      },
    },
    // Department response
    departmentResponse: {
      message: String,
      respondedBy: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "User",
      },
      respondedAt: Date,
    },
    // Resolution details
    resolvedAt: Date,
    rejectedAt: Date,
    rejectionReason: String,
  },
  {
    timestamps: true,
  }
);

// Index for faster queries
complaintSchema.index({ createdBy: 1, status: 1 });
complaintSchema.index({ assignedDepartment: 1, status: 1 });
complaintSchema.index({ createdAt: -1 });

const complaintModel = 
  mongoose.models.Complaint || 
  mongoose.model("Complaint", complaintSchema);

export default complaintModel;