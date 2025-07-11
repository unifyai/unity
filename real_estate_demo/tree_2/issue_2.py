from wizard_2 import RadioField

def get_issue_2_fields(ctx):
    match ctx["issue_element"]:
        case "Floors":
            options = ["Vinyl flooring is damaged", "Floorboard Broken and hole in the floor", "Floorboard loose but not broken", "Skirting board loose, rotten, or missing"]
            return [RadioField("issue_type", "Issue Location", options)]
        case "Walls":
            # TODO: Add specific fields for Walls issues
            options = ["Cracks in the wall", "Grouting between tiles missing or damaged", "Tiles are damaged or missing", "Grab rail (on wall) loose", "Skirting board loose, rotten, or missing", "Wall plaster damaged - loose crumbling or bulging", "Vent to outside wall missing or loose"]
            return [RadioField("issue_type", "Issue Location", options)]
        case "Ceilings":
            # TODO: Add specific fields for Ceilings issues
            options = ["Ceiling is falling down", "Cracks in the ceiling", "Ceiling plaster damaged - loose crumbling or bulging", "Roof leaking"]
            return [RadioField("issue_type", "Issue Location", options)]
        case "Kitchen Units":
            # TODO: Add specific fields for Kitchen Units issues
            pass
        case "Bath Stinks":
            # TODO: Add specific fields for Bath Stinks issues
            pass
        case "Toilets":
            # TODO: Add specific fields for Toilets issues
            pass
        case "Water Pipes":
            # TODO: Add specific fields for Water Pipes issues
            pass
        case "Taps":
            # TODO: Add specific fields for Taps issues
            pass
        case "Doors":
            # TODO: Add specific fields for Doors issues
            pass
        case "Locks":
            # TODO: Add specific fields for Locks issues
            pass
        case "Windows":
            # TODO: Add specific fields for Windows issues
            pass
        case "Lighting":
            # TODO: Add specific fields for Lighting issues
            options = ["No lights working in my property", "Some lights not working", "Light switch broken"]
            return [RadioField("issue_type", "Issue Location", options)]
        case "Other Electrics":
            # TODO: Add specific fields for Other Electrics issues
            pass
        case "Stair & Through Floor Lifts":
            # TODO: Add specific fields for Stair & Through Floor Lifts issues
            pass
        case "Alarms and Door Entry":
            # TODO: Add specific fields for Alarms and Door Entry issues
            pass
        case "Gas Heating & Hot Water":
            # TODO: Add specific fields for Gas Heating & Hot Water issues
            options = ["Gas boiler low pressure - no water leak", "Gas central heating not working", "Gas fire or heater not working", "Hot water not working", "Pipes have started making loud and unusual noises (I've not heard before)", "Error code on boiler display", "Gas fire or heater damaged", "Gas boiler leaking water on electrical fittings", "Gas boiler water leaking"]
            return [RadioField("issue_type", "Issue Location", options)]
        case "Electric / Storage Heating & Hot Water":
            # TODO: Add specific fields for Electric / Storage Heating & Hot Water issues
            pass
        case "Electric / Storage - Radiators":
            # TODO: Add specific fields for Electric / Storage - Radiators issues
            pass
        case "Gas - Radiators":
            # TODO: Add specific fields for Gas - Radiators issues
            pass
        case "Electric Showers":
            # TODO: Add specific fields for Electric Showers issues
            pass
        case "Fences":
            # TODO: Add specific fields for Fences issues
            options = ["Fence loose or falling down", "Fence panels missing or damaged", "Gate or gate post broken"]
            return [RadioField("issue_type", "Issue Location", options)]
        case "Brickwork":
            # TODO: Add specific fields for Brickwork issues
            pass
        case "Garage":
            # TODO: Add specific fields for Garage issues
            pass
        case "Groundworks":
            # TODO: Add specific fields for Groundworks issues
            pass
        case "Plumbing":
            # TODO: Add specific fields for Plumbing issues
            pass
        case "Roofing and tiles":
            # TODO: Add specific fields for Roofing and tiles issues
            pass
        case "Guttering":
            # TODO: Add specific fields for Guttering issues
            pass
        case "Aerials":
            # TODO: Add specific fields for Aerials issues
            pass