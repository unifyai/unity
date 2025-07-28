from wizard_2 import RadioField


def get_issue_2_fields(ctx):
    match ctx["issue_element"]:
        case "Floors":
            options = [
                "Vinyl flooring is damaged",
                "Floorboard Broken and hole in the floor",
                "Floorboard loose but not broken",
                "Skirting board loose, rotten, or missing",
            ]
            return [RadioField("issue_type", "Issue Type", options)]
        case "Walls":
            options = [
                "Cracks in the wall",
                "Grouting between tiles missing or damaged",
                "Tiles are damaged or missing",
                "Grab rail (on wall) loose",
                "Skirting board loose, rotten, or missing",
                "Wall plaster damaged - loose crumbling or bulging",
                "Vent to outside wall missing or loose",
            ]
            return [RadioField("issue_type", "Issue Type", options)]
        case "Ceilings":
            options = [
                "Ceiling is falling down",
                "Cracks in the ceiling",
                "Ceiling plaster damaged - loose crumbling or bulging",
                "Roof leaking",
            ]
            return [RadioField("issue_type", "Issue Type", options)]
        case "Stairs":
            options = [
                "Stair banister or spindles broken",
                "Staircase steps or floorboard broken",
            ]
            return [RadioField("issue_type", "Issue Type", options)]
        case "Kitchen Units":
            options = [
                "Kitchen wall unit loose or falling off the wall",
                "Kitchen unit doors or drawers loose or damaged",
                "Kitchen worktop damaged",
            ]
            return [RadioField("issue_type", "Issue Type", options)]
        case "Bath Sinks and Showers":
            options = [
                "Water leak",
                "No water in the property",
                "Blocked pipe or drain in hand basin or sink",
                "Sealant damaged around bath/basin or sink",
                "Shower curtain rail broken",
                "Mixer shower not working",
                "Bath is chipped",
                "Bath or basin waste trap plate has rusted",
            ]
            return [RadioField("issue_type", "Issue Type", options)]
        case "Toilets":
            options = [
                "Water leak",
                "My only toilet is blocked",
                "One of my toilets is blocked",
                "Toilet seat is broken",
                "Toilet is loose from the floor and I don't feel like I can use it",
                "Cistern behind toilet loose / unstable",
                "Broken pipe behind toilet",
                "Toilet is loose but I can still use it",
                "Toilet pan is cracked but not leaking",
            ]
            return [RadioField("issue_type", "Issue Type", options)]

        case "Water Pipes":
            options = [
                "Water leak",
                "No water in the property",
                "Pipes are frozen",
                "Blocked pipe or drain",
            ]
            return [RadioField("issue_type", "Issue Type", options)]

        case "Taps":
            options = ["No water in the property", "Tap leaking , loose or broken"]
            return [RadioField("issue_type", "Issue Type", options)]

        case "Doors":
            options = [
                "External door damaged and my property isn't secure / I can't lock it",
                "External Door is sticking / loose / draughty",
                "Door handle loose",
                "Fire door is sticking / loose / draughty",
                "Metal door closer broken",
                "Letterbox loose or broken",
                "Door frame split or damaged",
                "Internal door is sticking / loose / draughty",
            ]
            return [RadioField("issue_type", "Issue Type", options)]

        case "Locks":
            options = [
                "Door lock is sticking - my property is secure",
                "I want to change my locks",
                "I'm locked out",
            ]
            return [RadioField("issue_type", "Issue Type", options)]

        case "Windows":
            options = [
                "Smashed window",
                "Ground floor window doesn't close",
                "Upper floor window doesn't close",
                "Window is draughty",
                "Window frames or cills damaged",
                "Window hinges or handles broken",
                "Upstairs window restrictor loose/broken",
                "Window is misty between the panes of glass",
                "Window is cracked",
            ]
            return [RadioField("issue_type", "Issue Type", options)]

        case "Lighting":
            options = [
                "No lights working in my property",
                "Some lights not working",
                "Light switch broken",
            ]
            return [RadioField("issue_type", "Issue Type", options)]
        case "Other Electrics":
            options = [
                "No power in my property",
                "Electric in property keeps tripping or going off",
                "Switch or socket broken (sparking, making noises and smells like burning)",
                "Electrics need checking after a water leak",
                "Electric socket, switch or light fitting broken and the wires are exposed",
                "Electric socket, switch or light fitting broken wires aren't exposed",
                "Electric sockets aren't working but is not dangerous",
            ]
            return [RadioField("issue_type", "Issue Type", options)]

        case "Stair & Through Floor Lifts":
            options = [
                "Stair Lift is not working / moving",
                "Damage to stair lift (still working)",
                "Through floor lift not working",
                "Damage to through floor lift (still working)",
            ]
            return [RadioField("issue_type", "Issue Type", options)]

        case "Alarms and Door Entry":
            options = [
                "Door Entry system not working and door won't open",
                "Warden call system not working",
                "Burglar alarm system not working",
                "Fire alarm not working (panel broken, not sounding)",
                "Fire Door (linked to Alarm) not working",
            ]
            return [RadioField("issue_type", "Issue Type", options)]

        case "Gas Heating & Hot Water":
            options = [
                "Gas boiler low pressure - no water leak",
                "Gas central heating not working",
                "Gas fire or heater not working",
                "Hot water not working",
                "Pipes have started making loud and unusual noises (I've not heard before)",
                "Error code on boiler display",
                "Gas fire or heater damaged",
                "Gas boiler leaking water on electrical fittings",
                "Gas boiler water leaking",
            ]
            return [RadioField("issue_type", "Issue Type", options)]
        case "Electric / Storage Heating & Hot Water":
            options = [
                "No hot water from electric (immersion) heater",
                "An electric heater or fire is not working",
                "Hot Water Cylinder is Leaking",
                "Electric heating not working",
                "Air Source Heating - not working",
            ]
            return [RadioField("issue_type", "Issue Type", options)]

        case "Electric / Storage - Radiators":
            options = [
                "Radiator loose - coming away from the wall",
                "One of my radiators isn't getting warm",
                "Radiator needs putting back on the wall",
                "Radiator leaking - electrical fittings getting wet",
                "Radiator leaking - flooding the property",
                "Radiator leaking - collecting water in a bucket",
            ]
            return [RadioField("issue_type", "Issue Type", options)]

        case "Gas - Radiators":
            options = [
                "Radiator loose - coming away from the wall",
                "One of my radiators isn't getting warm",
                "Radiator needs putting back on the wall",
                "Radiator leaking - electrical fittings getting wet",
                "Radiator leaking - flooding the property",
                "Radiator leaking - collecting water in a bucket",
            ]
            return [RadioField("issue_type", "Issue Type", options)]
        case "Electric Showers":
            options = [
                "Electrical shower unit leaking",
                "Electric shower hose, handset or rail broken",
                "Electric shower not working / no hot water",
            ]
            return [RadioField("issue_type", "Issue Type", options)]

        # outside home
        case "Fences":
            # TODO: Add specific fields for Fences issues
            options = [
                "Fence loose or falling down",
                "Fence panels missing or damaged",
                "Gate or gate post broken",
            ]
            return [RadioField("issue_type", "Issue Type", options)]
        case "Brickwork":
            options = [
                "Fallen or unsafe wall",
                "Brick wall cracked",
                "Vent in wall to outside is missing or loose",
                "Damaged brickwork",
                "Outside wall covering (render) loose or cracked",
                "Outside wall covering (render) has fallen off",
            ]
            return [RadioField("issue_type", "Issue Type", options)]

        case "Garage":
            options = ["Garage door broken"]
            return [RadioField("issue_type", "Issue Type", options)]

        case "Groundworks":
            options = [
                "Driveway has multiple large cracks",
                "Garden path/step is loose/broken",
                "Personal rotary clothes line broken",
                "Retractable clothes line & post (in any area) broken",
                "Concrete post for personal clothes line broken",
            ]
            return [RadioField("issue_type", "Issue Type", options)]
        case "Plumbing":
            options = [
                "Pipes are frozen",
                "Blocked drain overflowing sewage",
                "Blocked drain",
                "Drain cover broken (no trip hazard)",
                "Drain cover missing (fall or trip hazard)",
                "Outside tap leaking or loose",
                "Outside tap dripping",
                "Gutter or drainpipe blocked or broken",
            ]
            return [RadioField("issue_type", "Issue Type", options)]
        case "Roofing and tiles":
            options = [
                "Tiles, lead, or chimney brickwork loose and dangerous",
                "Tiles, lead or chimney brickwork loose or fallen off roof",
                "Cracked roof tiles",
            ]
            return [RadioField("issue_type", "Issue Type", options)]
        case "Guttering":
            # TODO: Add specific fields for Guttering issues
            options = ["Gutter or drainpipe blocked or broken"]
            return [RadioField("issue_type", "Issue Type", options)]

        case "Aerials":
            options = ["Aerial or Satellite Dish not working"]
            return [RadioField("issue_type", "Issue Type", options)]
