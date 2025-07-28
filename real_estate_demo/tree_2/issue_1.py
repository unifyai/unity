from wizard_2 import *

def get_issue_1_fields(ctx):
    if ctx["area"] == "Inside Home":
        match(ctx["section"]):
            # inside home
            case "Floors, Walls, Ceilings and Stairs":
                options = ["Floors", "Walls", "Ceilings", "Stairs", "Kitchen Units"]
                return [RadioField("issue_element", "Issue Location", options)]
            case "Plumbing":
                options = ["Bath Sinks and Showers", "Toilets", "Water Pipes", "Taps"]
                return [RadioField("issue_element", "Issue Location", options)]
            case "Doors, Locks and Windows":
                options = ["Doors", "Locks", "Windows"]
                return [RadioField("issue_element", "Issue Location", options)]
            case "Electrics":
                options = ['Lighting', 'Other Electrics', 'Stair & Through Floor Lifts']
                return [RadioField("issue_element", "Issue Location", options)]
            case "Alarms and Door Entry":
                options = ["Alarms and Door Entry"]
                return [RadioField("issue_element", "Issue Location", options)]
            case "Heating & Hot Water":
                options = ['Gas Heating & Hot Water', 'Electric / Storage Heating & Hot Water', 'Electric / Storage - Radiators', 'Gas - Radiators', 'Electric Showers']
                return [RadioField("issue_element", "Issue Location", options)]

    # outside home
    match(ctx["section"]):
        case "Gardens and Fences":
            options = ['Fences', 'Brickwork', 'Garage', 'Groundworks']
            return [RadioField("issue_element", "Issue Location", options)]
        case "Plumbing":
            options = ["Plumbing"]
            return [RadioField("issue_element", "Issue Location", options)]
        case "Roofing":
            options = ["Roofing and tiles", "Guttering"]
            return [RadioField("issue_element", "Issue Location", options)]
        case "Electrics":
            options = ["Aerials", "Lighting"]
            return [RadioField("issue_element", "Issue Location", options)]
