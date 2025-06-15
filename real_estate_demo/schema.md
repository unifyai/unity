Below is a comprehensive schema detailing all the options and branches, including those shown in the screenshots and logical extensions for other repair types.

---

### **Comprehensive AI Schema & Decision Tree**

This schema is organized hierarchically, mirroring the user's journey from a general problem to a specific, actionable repair request.

#### **Phase 1: Pre-Screening & Triage**

This phase identifies the caller, assesses immediate danger, and performs an initial responsibility check.

*   `Caller_Profile`:
    *   `Tenant_Name` (string)
    *   `Tenant_Address` (string)
    *   `Is_Verified_Tenant` (boolean)
    *   `Vulnerability_Flag` (boolean) // Is there a pre-registered vulnerability?
*   `Triage_Assessment`:
    *   `Emergency_Status`: (enum: `Not_Emergency` | `Gas_Leak` | `Uncontrollable_Water_Leak` | `Electrical_Sparks_Fire` | `Structural_Collapse`) // Determines if the call needs immediate off-ramping to an emergency number.
    *   `Is_Tenant_Responsibility`: (boolean) // Determined by keyword analysis (e.g., "lightbulb," "paint," "toilet seat").

---

#### **Phase 2: Problem Diagnosis (The Full Decision Tree)**

This is the core of the AI's logic, a multi-tiered system to diagnose the repair.

**Tier 1: General Location**
*   `Location_General`:
    *   `Inside Your Home`
    *   `Outside Your Home`

---

**Tier 2: Primary System / Area**
*   `Area_Category`:
    *   *(If `Location_General` is `Inside Your Home`)*
        *   `Floors, Walls and Stairs`
        *   `Doors, Locks and Windows`
        *   `Plumbing`
        *   `Electrics`
        *   `Heating & Hot Water`
        *   `Kitchen Units & Worktops`
        *   `Alarms & Door Entry`
        *   `Other / Not Sure`
    *   *(If `Location_General` is `Outside Your Home`)*
        *   `Roof & Chimney`
        *   `External Walls & Brickwork`
        *   `Drains & Guttering`
        *   `Fences, Gates & Paths`
        *   `Communal Areas` (if applicable)

---

**Tier 3 -> Tier 4 -> Diagnostic Questions (The Detailed Breakdown)**

This is the most critical part. Each Tier 2 category branches into specific components (Tier 3), which then branch into specific issues (Tier 4), some of which trigger diagnostic questions.

---
**Branch: `Floors, Walls and Stairs`**
*   **Tier 3: `Ceilings`**
    *   **Tier 4: `Cracks in the ceiling`**
        *   **Diagnostic Question:** "To help us understand the severity, could you fit a Â£1 coin in the widest part of the crack?" (Yes/No)
    *   **Tier 4: `Ceiling plaster damaged - loose, crumbling or bulging`**
        *   **Diagnostic Question:** "Is the damaged area damp or showing signs of a water leak?" (Yes/No)
    *   **Tier 4: `Ceiling is falling down`**
        *   **Action:** Escalate priority. Skip diagnostics and go straight to scheduling.
*   **Tier 3: `Walls`**
    *   **Tier 4: `Cracks in internal wall`**
        *   **Diagnostic Question:** "Is the crack wider than the edge of a 10p coin?" (Yes/No)
    *   **Tier 4: `Damp or mould on wall`**
        *   **Diagnostic Question:** "Is the patch of damp larger than a dinner plate?" (Yes/No)
    *   **Tier 4: `Plaster damage`**
        *   **Diagnostic Question:** "Is the damaged area larger than your hand?" (Yes/No)
*   **Tier 3: `Floors`**
    *   **Tier 4: `Floorboards are loose or broken`**
        *   **Diagnostic Question:** "Does this pose an immediate trip hazard?" (Yes/No)
    *   **Tier 4: `Vinyl or laminate flooring is peeling or damaged`**
*   **Tier 3: `Stairs`**
    *   **Tier 4: `Handrail or bannister is loose`**
        *   **Action:** High priority.
    *   **Tier 4: `Stair tread is broken or loose`**
        *   **Action:** High priority.

---
**Branch: `Plumbing`**
*   **Tier 3: `Taps`** (e.g., Kitchen, Bathroom)
    *   **Tier 4: `Constantly dripping`**
    *   **Tier 4: `Can't be turned on/off`**
    *   **Tier 4: `Leaking from the base`**
*   **Tier 3: `Toilet`**
    *   **Tier 4: `Toilet won't flush`**
    *   **Tier 4: `Toilet cistern is constantly filling`**
    *   **Tier 4: `Leaking from the toilet base or cistern`**
        *   **Diagnostic Question:** "Is the leak contained or is water spreading across the floor?" (Contained/Spreading)
*   **Tier 3: `Pipes`**
    *   **Tier 4: `Visible leak from a pipe`**
        *   **Diagnostic Question:** "Is it a slow drip or a steady flow of water?" (Drip/Flow) -> If `Flow`, escalate to emergency check.
    *   **Tier 4: `Blocked sink or bath`**
        *   **Diagnostic Question:** "Have you already tried a standard, off-the-shelf drain unblocker?" (Yes/No)
*   **Tier 3: `Shower`**
    *   **Tier 4: `No water from shower`**
    *   **Tier 4: `Leaking shower head or hose`**
    *   **Tier 4: `Water leaking from the tray or enclosure`**

---
**Branch: `Heating & Hot Water`**
*   **Tier 3: `Radiators`**
    *   **Tier 4: `Radiator is cold`**
        *   **Diagnostic Question:** "Are other radiators in the house working correctly?" (Yes/No)
        *   **Diagnostic Question 2:** "Have you tried bleeding the radiator?" (Yes/No/Not Sure How)
    *   **Tier 4: `Radiator is leaking`**
        *   **Diagnostic Question:** "Where is it leaking from? The valve, the body of the radiator, or the pipe?" (Valve/Body/Pipe)
*   **Tier 3: `Boiler / Hot Water System`**
    *   **Tier 4: `No heating and no hot water`**
        *   **Diagnostic Question:** "Is your boiler showing an error code or a flashing light? If so, what is it?" (User provides code/description)
        *   **Diagnostic Question 2:** "Have you checked if your gas and electricity supply are both on?" (Yes/No)
    *   **Tier 4: `No hot water, but heating is working`**
    *   **Tier 4: `Banging or loud noises from boiler`**

---
**Branch: `Electrics`**
*   **Tier 3: `Sockets & Switches`**
    *   **Tier 4: `Socket not working`**
        *   **Diagnostic Question:** "Have you checked the trip switch in your fuse box?" (Yes/No/Not Sure How)
    *   **Tier 4: `Socket is cracked, burnt or buzzing`**
        *   **Action:** Escalate to emergency check. "Please do not touch the socket. Is there any sign of smoke or sparks?"
*   **Tier 3: `Lighting`**
    *   **Tier 4: `A single light fitting is not working`**
        *   **Diagnostic Question:** "Just to be sure, have you already tried changing the lightbulb?" (Yes/No) -> If No, potential tenant responsibility.
    *   **Tier 4: `All lights in one room are out`**
    *   **Tier 4: `All lights in the property are out`**
        *   **Diagnostic Question:** "Are your neighbours' lights on? This will help us know if it's a power cut in your area." (Yes/No/Not Sure)
*   **Tier 3: `Other Electrical Items`**
    *   **Tier 4: `Electric shower not working / not heating`**
    *   **Tier 4: `Extractor fan not working`**

---

**Tier 5: Precise Location (Room)**
*   `Location_Specific_Room`: (enum)
    *   `Attic / Loft`
    *   `Bathroom`
    *   `Bedroom (Main)`
    *   `Bedroom (Other)`
    *   `Cellar / Basement`
    *   `Dining Room`
    *   `Hallway`
    *   `Kitchen`
    *   `Landing`
    *   `Laundry Room`
    *   `Living Room / Lounge`
    *   `Toilet (Downstairs)`
    *   `Multiple Rooms`
    *   `Not Applicable` (for outside jobs)

---
**Tier 6: Final Details & Confirmation**
*   `User_Description_Transcript`: (string) // "Is there anything else you think we might need to know about this repair?"
*   `Agreements`:
    *   `Data_Accuracy_Confirmation`: (boolean) // "I have given as much information as I can..."
    *   `Adult_Over_18_Confirmation`: (boolean) // "I understand someone over 18 must be present..."

---

#### **Phase 3: Logistics & Scheduling**

*   `Appointment`:
    *   `Available_Slots`: (array of objects) `[{date: "YYYY-MM-DD", time_slot: "AM 08:00-13:00"}, ...]`
    *   `Selected_Slot`: (object)
*   `Access_Details`:
    *   `Confirmation_Contact_Number`: (string)
    *   `Access_Notes`: (string) // "Are there any special instructions for our operative, like a key safe code or to be aware of a pet?"

---

#### **Phase 4: Confirmation & Close-Out**

*   `Repair_Summary`: (A consolidated object of all collected data for final confirmation)
    *   `Summary_Location`: "Inside Your Home in the Kitchen"
    *   `Summary_Type`: "Plumbing"
    *   `Summary_Issue`: "Toilet - Leaking from the base"
    *   `Summary_Appointment`: "Tuesday 11th Feb, 8 AM to 1 PM"
*   `Repair_Reference_Number`: (string, generated by system)
*   `Next_Steps_Information`: (string) // e.g., "You will now receive an SMS confirmation..."