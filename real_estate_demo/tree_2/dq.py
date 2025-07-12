from wizard_2 import *

dq_options = ["Vinyl flooring is damaged",
     "Floorboard Broken and hole in the floor",
     "Cracks in the wall",
     "Grouting between tiles missing or damaged",
     "Tiles are damaged or missing",
     "Ceiling is falling down",
     "Cracks in the ceiling",
     "Ceiling plaster damaged - loose crumbling or bulging",
     "Roof leaking",
     "No lights working in my property",
     "Some lights not working",
     "Gas boiler low pressure - no water leak",
     "Gas central heating not working",
     "Gas fire or heater not working",
     "Fence panels missing or damaged"]




def get_dq_fields(ctx):
    match ctx["issue_type"]:
        # Cases for 'Floors' sub-issues
        case "Vinyl flooring is damaged":
            return [RadioField("is_trip_hazard", "Is it a trip hazard", ["yes", "no"])]
        case "Floorboard Broken and hole in the floor":
            return [RadioField("is_trip_hazard", "Is it a trip hazard", ["yes", "no"])]
        # Cases for 'Walls' sub-issues
        case "Cracks in the wall":
            return [RadioField("euro_coin", "Can you fit a one euro coin in the gap?", ["yes", "no"])]
        case "Grouting between tiles missing or damaged":
            return [RadioField("tiles_fitted", "Were the tiles fitted by Midland Heart?", ["yes", "no"])]
        case "Tiles are damaged or missing":
            return [RadioField("tiles_fitted", "Were the tiles fitted by Midland Heart?", ["yes", "no"])]
        # Cases for 'Ceilings' sub-issues
        case "Ceiling is falling down":
            return [RadioField("is_dangerous", "Is it dangerous?", ["yes", "no"])]
        case "Cracks in the ceiling":
            return [RadioField("euro_coin", "Can you fit a one euro coin in the gap?", ["yes", "no"])]
        case "Ceiling plaster damaged - loose crumbling or bulging":
            # TODO: Add specific fields for Ceiling plaster damaged - loose crumbling or bulging
            return [RadioField("ceiling_about_to_fall", "Do you think the ceiling is about to fall down?", ["yes", "no"])]

        case "Roof leaking":
            # TODO: Add specific fields for Roof leaking
            return [RadioField("catch_water_in_bucket", "Can you catch the water in a bucket or other container?", ["yes", "no"])]

        # Cases for 'Lighting' sub-issues
        case "No lights working in my property":
            return [RadioField("checked_trip_switch", "Have you checked the trip switch?", ["yes", "no"])]

        case "Some lights not working":
            return [RadioField("any_lights_working", "Are there any lights working in the location?", ["yes", "no"])]

        # Cases for 'Gas Heating & Hot Water' sub-issues
        case "Gas boiler low pressure - no water leak":
            return [RadioField("re_pressurise_boiler", "Have you tried to re-pressurise the boiler?", ["yes", "no"])]
        case "Gas central heating not working":
            return [RadioField("checked_gas_meter", "Have you checked your gas meter is topped up?", ["yes", "no"])]
        case "Gas fire or heater not working":
            return [RadioField("only_heating_form", "Is this your only form of heating?", ["yes", "no"])]
        case "Hot water not working":
            return [RadioField("checked_gas_meter", "Have you checked your gas meter is topped up?", ["yes", "no"])]

        case "Pipes have started making loud and unusual noises (I've not heard before)":
            # TODO: Add specific fields for Pipes have started making loud and unusual noises
            pass
        case "Error code on boiler display":
            # TODO: Add specific fields for Error code on boiler display
            pass
        case "Gas fire or heater damaged":
            # TODO: Add specific fields for Gas fire or heater damaged
            pass
        case "Gas boiler leaking water on electrical fittings":
            # TODO: Add specific fields for Gas boiler leaking water on electrical fittings
            pass
        case "Gas boiler water leaking":
            # TODO: Add specific fields for Gas boiler water leaking
            pass

        # Cases for 'Fences' sub-issues
        case "Fence loose or falling down":
            # TODO: Add specific fields for Fence loose or falling down
            pass
        case "Fence panels missing or damaged":
            # TODO: Add specific fields for Fence panels missing or damaged
            return [RadioField("fence_location", "Where is the fence?", ["Next to a public footpath or road", "Between your property and neighbour"])]

        case "Gate or gate post broken":
            # TODO: Add specific fields for Gate or gate post broken
            pass


def handle_dq_next(ctx):
    match ctx["issue_type"]:
        # Cases for 'Floors' sub-issues
        case "Vinyl flooring is damaged":
            if ctx["is_trip_hazard"] == "yes":
                return "more_info"
            return "location" # only two locations show up here (bathroom, kitchen)
        
        case "Floorboard Broken and hole in the floor":
            if ctx["is_trip_hazard"] == "yes":
                return "more_info"
            return "location"
        # Cases for 'Walls' sub-issues
        case "Cracks in the wall":
            if ctx["euro_coin"] == "yes":
                return "location"
            return "diy"
        case "Grouting between tiles missing or damaged":
            if ctx["tiles_fitted"]:
                return "location" # bathroom, kitchen, laundry
            return "diy"
        case "Tiles are damaged or missing":
            if ctx["tiles_fitted"]:
                return "location" # bathroom, kitchen, laundry
            return "diy"
        # Cases for 'Ceilings' sub-issues
        case "Ceiling is falling down":
            if ctx["is_dangerous"] == "yes":
                return "more_info"
            return "location"
        case "Cracks in the ceiling":
            if ctx["euro_coin"] == "yes":
                return "location"
            return "diy"
        case "Ceiling plaster damaged - loose crumbling or bulging":
            if ctx["ceiling_about_to_fall"] == "no":
                return "location"
            return "more_info"

        case "Roof leaking":
            if ctx["catch_water_in_bucket"]:
                return "location"
            return "diy"

        case "No lights working in my property":
            if ctx["checked_trip_switch"] == "yes":
                return "more_info"
            return "diy"
        
        case "Some lights not working":
            if ctx["any_lights_working"] == "yes":
                return "location"
            return "more_info"

        # Cases for 'Gas Heating & Hot Water' sub-issues
        case "Gas boiler low pressure - no water leak":
            if ctx["re_pressurise_boiler"] == "yes":
                return "more_info"
            return "diy"
        case "Gas central heating not working":
            if ctx["checked_gas_meter"] == "yes":
                return "more_info"
            return "diy"
        # raises another diagnostic question
        case "Gas fire or heater not working":
            if ctx["only_heating_form"] == "yes":
                return "more_info"
            return "diagnostic_question_2"
        case "Hot water not working":
            if ctx["checked_gas_meter"] == "yes":
                return "diagnostic_question_2"
            return "diy"

        case "Pipes have started making loud and unusual noises (I've not heard before)":
            # TODO: Add specific fields for Pipes have started making loud and unusual noises
            pass
        case "Error code on boiler display":
            # TODO: Add specific fields for Error code on boiler display
            pass
        case "Gas fire or heater damaged":
            # TODO: Add specific fields for Gas fire or heater damaged
            pass
        case "Gas boiler leaking water on electrical fittings":
            # TODO: Add specific fields for Gas boiler leaking water on electrical fittings
            pass
        case "Gas boiler water leaking":
            # TODO: Add specific fields for Gas boiler water leaking
            pass

        # Cases for 'Fences' sub-issues
        case "Fence loose or falling down":
            # TODO: Add specific fields for Fence loose or falling down
            pass
        case "Fence panels missing or damaged":
            # TODO: Add specific fields for Fence panels missing or damaged
            if ctx["fence_location"] == "Next to a public footpath or road":
                return "location"
            return "diy"

        case "Gate or gate post broken":
            # TODO: Add specific fields for Gate or gate post broken
            pass