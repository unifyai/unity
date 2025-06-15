Each flow includes `(Internal AI Logic)` annotations to explain what the AI is processing and how it's making decisions based on the schema, without exposing that technical logic to the user.

---

### **Flow 1: The Standard, Guided Conversation**

**Scenario:** A user who is happy to be led step-by-step through the process. The AI follows the schema's hierarchy perfectly.

**AI Agent:** "Hello, you're through to the Midland Heart repairs line. To get started, could you please tell me the first line of your address and your postcode?"
**User:** "Hi, it's 14 Orchard Way, B4 5FG."
**AI Agent:** "Thank you. I've found your account for Ms. Anya Sharma at that address. Is that you?"
**User:** "Yes, it is."

**(Internal AI Logic: `Caller_Profile` is now verified. Proceed to Triage.)**

**AI Agent:** "Great, Anya. Just a quick safety question before we go on: is your repair an emergency, like a major water leak you can't control, a smell of gas, or sparking electrics?"
**User:** "No, nothing like that."

**(Internal AI Logic: `Triage_Assessment` -> `Emergency_Status` set to `Not_Emergency`. Proceed to Diagnosis Tier 1.)**

**AI Agent:** "Okay, good. To make sure I understand the problem correctly, is the repair needed inside your home, or outside?"
**User:** "It's inside."

**(Internal AI Logic: `Location_General` set to `Inside Your Home`. Proceed to Tier 2.)**

**AI Agent:** "Inside, got it. And what general area does it relate to? For example, Heating and Hot Water, Plumbing, or maybe Doors and Windows?"
**User:** "It's the heating."

**(Internal AI Logic: `Area_Category` set to `Heating & Hot Water`. Proceed to Tier 3.)**

**AI Agent:** "Okay, the heating. Is the issue with a radiator, or with the boiler and hot water system itself?"
**User:** "It's one of the radiators."

**(Internal AI Logic: `Tier 3` set to `Radiators`. Proceed to Tier 4.)**

**AI Agent:** "And what's happening with the radiator? Is it cold when the heating is on, or is it leaking?"
**User:** "It's cold. The others are all hot, but this one's stone cold."

**(Internal AI Logic: `Tier 4` set to `Radiator is cold`. This triggers a diagnostic question from the schema.)**

**AI Agent:** "I see. That's a common issue. Just to help our engineer prepare, have you already tried to bleed that radiator yourself?"
**User:** "No, I wouldn't know how to do that."

**(Internal AI Logic: `Diagnostic_Response` captured. Proceed to Tier 5.)**

**AI Agent:** "No problem at all, that's what we're here for. Which room is the cold radiator in?"
**User:** "It's in the main bedroom."
**AI Agent:** "Okay, main bedroom. I've got everything I need to log that. Let me look for appointments... I have slots available this Friday. Would you prefer a morning slot, 8 AM to 1 PM, or an afternoon slot, 12 PM to 5 PM?"
**User:** "Morning, please."
**AI Agent:** "Perfect. I've booked that for you. So just to confirm, we have a heating engineer coming to fix a **cold radiator** in the **main bedroom** this **Friday between 8 AM and 1 PM**. Can you also confirm someone over 18 will be at home?"
**User:** "Yes, that's all correct. I'll be here."
**AI Agent:** "Wonderful. Your repair reference is H-45-B21. You'll get a text confirming these details in a moment. Is there anything else I can help with today?"
**User:** "No, that's great. Thank you."

---

### **Flow 2: The "Informed User" - Skipping Ahead**

**Scenario:** A user who knows exactly what they want and provides the information upfront. The AI adapts, confirms its understanding, and skips the redundant questions.

**AI Agent:** "Hello, you're through to the Midland Heart repairs line. How can I help you today?"
**User:** "Hi there, I need to book a repair. The lock on my front door is broken; the key just spins around."

**(Internal AI Logic: The AI parses the user's statement and pre-fills the schema. `Location_General: Inside Your Home`, `Area_Category: Doors, Locks and Windows`, `Tier 3: Doors`, `Tier 4: Lock is broken`. It recognizes it still needs to identify the caller.)**

**AI Agent:** "Okay, a broken front door lock. I can certainly get that sorted for you. First, so I can pull up your details, could you tell me your address?"
**User:** "14 Orchard Way, B4 5FG."
**AI Agent:** "Thanks, Anya. I've got your details. So, I've logged the issue as a broken lock on your front door. Is there anything else our operative should know about the problem?"
**User:** "No, that's it. It just won't lock."

**(Internal AI Logic: The AI has all the diagnostic information it needs. It can skip the entire Tier 2-4 questioning process and jump straight to scheduling.)**

**AI Agent:** "Understood. Because this relates to securing your home, we treat it as a high priority. Let me see what we have... I can get someone to you this afternoon between 1 PM and 5 PM, or tomorrow morning between 8 AM and 1 PM. What works best for you?"
**User:** "This afternoon would be amazing, thank you."

**(The flow continues to the confirmation stage.)**

---

### **Flow 3: The "Correction" Flow - Changing Information**

**Scenario:** The user makes a mistake and corrects themselves later in the conversation. The AI handles this gracefully.

**AI Agent:** "...And which room has the damaged plaster?"
**User:** "It's in the living room."
**AI Agent:** "Okay, living room. Is the damaged area larger than the palm of your hand?"
**User:** "Yes, it is."

**(The AI proceeds through the flow and gets to the final summary.)**

**AI Agent:** "Okay, Anya, let's just confirm everything. I'm booking a repair for **damaged plaster** in the **Living Room**, for next Tuesday in the morning slot. Is that all correct?"
**User:** "Oh, wait, I'm so sorry, I said living room but I meant the hallway. My mistake. It's the wall in the hallway."

**(Internal AI Logic: The user has provided new information that overrides a previous data point. The AI must update the `Location_Specific_Room` in the schema from `Living Room` to `Hallway` and then re-confirm.)**

**AI Agent:** "Not a problem at all, thank you for clarifying! Let me just update that for you... okay, so I've changed the location to the **Hallway**. So, that's a repair for damaged plaster in the Hallway, next Tuesday morning. Does that sound right now?"
**User:** "Yes, that's perfect. Sorry about that."
**AI Agent:** "Don't worry about it! It's all confirmed for you now..."

---

### **Flow 4: The "Ambiguous Problem" - Collaborative Diagnosis**

**Scenario:** The user is unsure how to categorize their problem. The AI helps them figure it out by asking intelligent questions.

**AI Agent:** "...And what general area does it relate to? For example, Heating, Plumbing, or Electrics?"
**User:** "Umm, I'm not really sure. The light in the bathroom is flickering, but the extractor fan has also stopped working at the same time."

**(Internal AI Logic: The user has described two related issues (`flickering light`, `fan not working`) that fall under `Electrics`. The AI should group them rather than treating them as separate repairs.)**

**AI Agent:** "Okay, thanks for explaining that. It sounds like there might be an electrical issue in the bathroom affecting both the light and the fan. Does that sound right to you?"
**User:** "Yes, I think so. They stopped working together."
**AI Agent:** "Right. Just to be safe, have you tried checking the trip switch in your fuse box to see if anything has switched off?"
**User:** "I did look, but nothing seemed to be off."

**(Internal AI Logic: The AI has successfully guided the user to the `Electrics` category and performed a basic diagnostic. It can now log the job with a more detailed description for the engineer.)**

**AI Agent:** "Okay, that's helpful. I'll log this as an 'Electrical Fault' in the bathroom, and I'll make a note for the electrician that it's affecting both the light fitting and the extractor fan. Now, let's find an appointment for you..."

---

### **Flow 5: The "Tenant Responsibility" - Graceful Off-Ramp**

**Scenario:** The user requests a repair that is not the landlord's responsibility. The AI must decline the request helpfully and politely.

**AI Agent:** "Hello, you're through to the Midland Heart repairs line. How can I help?"
**User:** "Hi, I've got a blocked sink in the kitchen."

**(Internal AI Logic: The AI recognizes `Blocked sink` as a potential tenant responsibility, which triggers a diagnostic question from the schema.)**

**AI Agent:** "Okay, a blocked kitchen sink. I can help with that. Can I just ask, have you already tried using a standard, off-the-shelf drain unblocking product?"
**User:** "No, I haven't. I just thought you guys would send a plumber."
**AI Agent:** "I understand. For simple blockages like a sink, we do ask that residents first try a standard unblocker, as this often solves the problem. This is part of the tenant responsibilities outlined in your tenancy handbook. If you try that and it's still blocked, it could be a deeper issue, and you should definitely call us back."
**User:** "Oh, okay. I didn't know. I can try that first then."
**AI Agent:** "Great. Is there any other, different repair issue I can help you with today?"
**User:** "No, that was it."
**AI Agent:** "Alright. Well, I hope that works for you, and please don't hesitate to call back if it doesn't. Thanks for calling."