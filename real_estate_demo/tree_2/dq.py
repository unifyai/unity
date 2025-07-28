from wizard_2 import *

dq_options = [
    # Floors
    "Vinyl flooring is damaged",
    "Floorboard Broken and hole in the floor",
    # Walls
    "Cracks in the wall",
    "Grouting between tiles missing or damaged",
    "Tiles are damaged or missing",
    # Ceilings
    "Ceiling is falling down",
    "Cracks in the ceiling",
    "Ceiling plaster damaged - loose crumbling or bulging",
    "Roof leaking",
    # Stairs
    "Stair banister or spindles broken",
    # Kitchen Units
    "Kitchen wall unit loose or falling off the wall",
    # Bath, Sinks & Showers
    "Water leak",
    "Sealant damaged around bath/basin or sink",
    "Shower curtain rail broken",
    "Mixer shower not working",
    # Toilets
    "My only toilet is blocked",
    "One of my toilets is blocked",
    "Toilet is loose from the floor and I don't feel like I can use it",
    "Cistern behind toilet loose / unstable",
    "Broken pipe behind toilet",
    # Doors & Locks
    "Fire door is sticking / loose / draughty",
    "Door frame split or damaged",
    "Door lock is sticking - my property is secure",
    # Windows
    "Smashed window",
    # Lighting & Electrics
    "No lights working in my property",
    "Some lights not working",
    "No power in my property",
    # Stair & Through-floor Lifts
    "Stair Lift is not working / moving",
    "Damage to stair lift (still working)",
    "Through floor lift not working",
    "Damage to through floor lift (still working)",
    # Gas Heating & Hot Water
    "Gas boiler low pressure - no water leak",
    "Gas central heating not working",
    "Gas fire or heater not working",
    "Hot water not working",
    # Fences
    "Fence panels missing or damaged",
    # Brickwork
    "Brick wall cracked",
    "Outside wall covering (render) loose or cracked",
]


def get_dq_fields(ctx):
    match ctx["issue_type"]:
        # Floors
        case "Vinyl flooring is damaged":
            return [RadioField("is_trip_hazard", "Is it a trip hazard", ["yes", "no"])]
        case "Floorboard Broken and hole in the floor":
            return [RadioField("is_trip_hazard", "Is it a trip hazard", ["yes", "no"])]

        # Walls
        case "Cracks in the wall":
            return [
                RadioField(
                    "euro_coin",
                    "Can you fit a one euro coin in the gap?",
                    ["yes", "no"],
                ),
            ]
        case "Grouting between tiles missing or damaged":
            return [
                RadioField(
                    "tiles_fitted",
                    "Were the tiles fitted by Midland Heart?",
                    ["yes", "no"],
                ),
            ]
        case "Tiles are damaged or missing":
            return [
                RadioField(
                    "tiles_fitted",
                    "Were the tiles fitted by Midland Heart?",
                    ["yes", "no"],
                ),
            ]

        # Ceilings
        case "Ceiling is falling down":
            return [RadioField("is_dangerous", "Is it dangerous?", ["yes", "no"])]
        case "Cracks in the ceiling":
            return [
                RadioField(
                    "euro_coin",
                    "Can you fit a one euro coin in the gap?",
                    ["yes", "no"],
                ),
            ]
        case "Ceiling plaster damaged - loose crumbling or bulging":
            return [
                RadioField(
                    "ceiling_about_to_fall",
                    "Do you think the ceiling is about to fall down?",
                    ["yes", "no"],
                ),
            ]
        case "Roof leaking":
            return [
                RadioField(
                    "catch_water_in_bucket",
                    "Can you catch the water in a bucket or other container?",
                    ["yes", "no"],
                ),
            ]

        # Stairs
        case "Stair banister or spindles broken":
            return [
                RadioField(
                    "broken_or_missing",
                    "Is it broken off or missing?",
                    ["yes", "no"],
                ),
            ]

        # Kitchen Units
        case "Kitchen wall unit loose or falling off the wall":
            return [
                RadioField(
                    "falling_danger",
                    "Are they in danger of falling off and causing harm?",
                    ["yes", "no"],
                ),
            ]
        # diy
        # case "Kitchen unit doors or drawers loose or damaged":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]

        # Bath, Sinks & Showers
        case "Water leak":
            return [
                RadioField(
                    "catch_water_in_bucket",
                    "Can you catch the water in a bucket or other container?",
                    ["yes", "no"],
                ),
            ]

        # location
        # case "No water in the property":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]

        # diy
        # case "Blocked pipe or drain in hand basin or sink":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]

        case "Sealant damaged around bath/basin or sink":
            return [
                RadioField(
                    "black_mold",
                    "Is there black mold on the sealant?",
                    ["yes", "no"],
                ),
            ]

        case "Shower curtain rail broken":
            return [
                RadioField(
                    "fitted_by_midland",
                    "Was it fitted by midland?",
                    ["yes", "no"],
                ),
            ]
        case "Mixer shower not working":
            return [
                RadioField(
                    "fitted_by_midland",
                    "Was it fitted by midland?",
                    ["yes", "no"],
                ),
            ]
        # diy
        # case "Bath is chipped":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]

        # Toilets
        case "My only toilet is blocked":
            return [
                RadioField(
                    "unblock_yourself",
                    "Have you tried to unblock it yourself?",
                    ["yes", "no"],
                ),
            ]

        case "One of my toilets is blocked":
            return [
                RadioField(
                    "unblock_yourself",
                    "Have you tried to unblock it yourself?",
                    ["yes", "no"],
                ),
            ]
        # diy
        # case "Toilet seat is broken":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        case "Toilet is loose from the floor and I don't feel like I can use it":
            return [
                RadioField("only_toilet", "Is this your only toilet?", ["yes", "no"]),
            ]
        case "Cistern behind toilet loose / unstable":
            return [
                RadioField("only_toilet", "Is this your only toilet?", ["yes", "no"]),
            ]
        case "Broken pipe behind toilet":
            return [
                RadioField("only_toilet", "Is this your only toilet?", ["yes", "no"]),
            ]

        # Water Pipes
        # diy
        # case "Pipes are frozen":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # diy
        # case "Blocked pipe or drain":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]

        # Taps -> location (bathroom, kitchen, laundry room)
        # case "Tap leaking , loose or broken":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]

        # Doors
        # case "External door damaged and my property isn't secure / I can't lock it":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "External Door is sticking / loose / draughty":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Door handle loose":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        case "Fire door is sticking / loose / draughty":
            return [
                RadioField(
                    "pass_smoke",
                    "Does the door have damage that would allow the passage of smoke or fire from the flat to the communal area",
                    ["yes", "no"],
                ),
            ]
        case "Door frame split or damaged":
            return [
                RadioField(
                    "door_open_inside_corridor",
                    "Does your door open onto the inside corridor of a block of flats?",
                    ["yes", "no"],
                ),
            ]
        # diy
        # case "Internal door is sticking / loose / draughty":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]

        # Locks
        case "Door lock is sticking - my property is secure":
            return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # diy
        # case "I want to change my locks":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "I'm locked out":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]

        # Windows
        case "Smashed window":
            return [
                RadioField(
                    "crime_ref_num",
                    "Do you have a crime reference number?",
                    ["yes", "no"],
                ),
            ]

        # Lighting
        case "No lights working in my property":
            return [
                RadioField(
                    "checked_trip_switch",
                    "Have you checked the trip switch?",
                    ["yes", "no"],
                ),
            ]
        case "Some lights not working":
            return [
                RadioField(
                    "any_lights_working",
                    "Are there any lights working in the location?",
                    ["yes", "no"],
                ),
            ]
        # Other Electrics
        case "No power in my property":
            return [
                RadioField(
                    "checked_trip_switch",
                    "Have you checked the trip switch?",
                    ["yes", "no"],
                ),
            ]
        # Stair & Through-floor Lifts
        case "Stair Lift is not working / moving":
            return [
                RadioField(
                    "fitted_by_midland",
                    "Was it fitted by midland?",
                    ["yes", "no"],
                ),
            ]
        case "Damage to stair lift (still working)":
            return [
                RadioField(
                    "fitted_by_midland",
                    "Was it fitted by midland?",
                    ["yes", "no"],
                ),
            ]
        case "Through floor lift not working":
            return [
                RadioField(
                    "fitted_by_midland",
                    "Was it fitted by midland?",
                    ["yes", "no"],
                ),
            ]
        case "Damage to through floor lift (still working)":
            return [
                RadioField(
                    "fitted_by_midland",
                    "Was it fitted by midland?",
                    ["yes", "no"],
                ),
            ]

        # Alarms & Door Entry
        # case "Door Entry system not working and door won't open":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Warden call system not working":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Burglar alarm system not working":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Fire alarm not working (panel broken, not sounding)":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Fire Door (linked to Alarm) not working":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]

        # Gas Heating & Hot Water
        case "Gas boiler low pressure - no water leak":
            return [
                RadioField(
                    "re_pressurise_boiler",
                    "Have you tried to re-pressurise the boiler?",
                    ["yes", "no"],
                ),
            ]
        case "Gas central heating not working":
            return [
                RadioField(
                    "checked_gas_meter",
                    "Have you checked your gas meter is topped up?",
                    ["yes", "no"],
                ),
            ]
        case "Gas fire or heater not working":
            return [
                RadioField(
                    "only_heating_form",
                    "Is this your only form of heating?",
                    ["yes", "no"],
                ),
            ]
        case "Hot water not working":
            return [
                RadioField(
                    "checked_gas_meter",
                    "Have you checked your gas meter is topped up?",
                    ["yes", "no"],
                ),
            ]
        # case "Pipes have started making loud and unusual noises (I've not heard before)":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Error code on boiler display":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Gas fire or heater damaged":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Gas boiler leaking water on electrical fittings":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Gas boiler water leaking":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]

        # Electric / Storage Heating & Hot Water
        # case "No hot water from electric (immersion) heater":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "An electric heater or fire is not working":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Hot Water Cylinder is Leaking":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Electric heating not working":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Air Source Heating - not working":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]

        # Electric / Storage – Radiators (and Gas Radiators share same issue-texts)
        # case "Radiator loose - coming away from the wall":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "One of my radiators isn't getting warm":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Radiator needs putting back on the wall":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Radiator leaking - electrical fittings getting wet":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Radiator leaking - flooding the property":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Radiator leaking - collecting water in a bucket":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]

        # Electric Showers
        # case "Electrical shower unit leaking":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Electric shower hose, handset or rail broken":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Electric shower not working / no hot water":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]

        # Fences
        case "Fence panels missing or damaged":
            return [
                RadioField(
                    "fence_location",
                    "Where is the fence?",
                    [
                        "Next to a public footpath or road",
                        "Between your property and neighbour",
                    ],
                ),
            ]

        # Brickwork
        case "Brick wall cracked":
            return [
                RadioField(
                    "euro_coin",
                    "Can you fit a one euro coin in the gap?",
                    ["yes", "no"],
                ),
            ]
        case "Outside wall covering (render) loose or cracked":
            return [
                RadioField(
                    "euro_coin",
                    "Can you fit a one euro coin in the gap?",
                    ["yes", "no"],
                ),
            ]

        # Garage
        # case "Garage door broken":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]

        # Groundworks
        # case "Driveway has multiple large cracks":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Garden path/step is loose/broken":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Personal rotary clothes line broken":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Retractable clothes line & post (in any area) broken":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Concrete post for personal clothes line broken":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]

        # Plumbing
        # case "Blocked drain overflowing sewage":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Blocked drain":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Drain cover broken (no trip hazard)":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Drain cover missing (fall or trip hazard)":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Outside tap leaking or loose":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Outside tap dripping":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Gutter or drainpipe blocked or broken":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]

        # Roofing & Tiles
        # case "Tiles, lead, or chimney brickwork loose and dangerous":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Tiles, lead or chimney brickwork loose or fallen off roof":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]
        # case "Cracked roof tiles":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]

        # Guttering
        # case "Gutter or drainpipe blocked or broken":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]

        # Aerials
        # case "Aerial or Satellite Dish not working":
        #     return [RadioField("placeholder", "Placeholder question", ["yes", "no"])]


def handle_dq_next(ctx):
    match ctx["issue_type"]:
        # ───────────────────────────── Floors ─────────────────────────────
        case "Vinyl flooring is damaged":
            return "location" if ctx["is_trip_hazard"] == "yes" else "location"

        case "Floorboard Broken and hole in the floor":
            return "location" if ctx["is_trip_hazard"] == "yes" else "location"

        # ───────────────────────────── Walls ──────────────────────────────
        case (
            "Cracks in the wall"
            | "Brick wall cracked"
            | "Outside wall covering (render) loose or cracked"
        ):
            return "location" if ctx["euro_coin"] == "yes" else "diy"

        case (
            "Grouting between tiles missing or damaged" | "Tiles are damaged or missing"
        ):
            return "location" if ctx["tiles_fitted"] == "yes" else "diy"

        # ──────────────────────────── Ceilings ────────────────────────────
        case "Ceiling is falling down":
            return "location" if ctx["is_dangerous"] == "yes" else "location"

        case "Cracks in the ceiling":
            return "location" if ctx["euro_coin"] == "yes" else "diy"

        case "Ceiling plaster damaged - loose crumbling or bulging":
            return "location" if ctx["ceiling_about_to_fall"] == "yes" else "location"

        case "Roof leaking":
            return "location" if ctx["catch_water_in_bucket"] == "yes" else "diy"

        # ───────────────────────────── Stairs ─────────────────────────────
        case "Stair banister or spindles broken":
            return "location" if ctx["broken_or_missing"] == "yes" else "diy"

        # ───────────────────────── Kitchen units ──────────────────────────
        case "Kitchen wall unit loose or falling off the wall":
            return "location" if ctx["falling_danger"] == "yes" else "diy"

        # ─────────────────────── Bath / Sinks / Showers ───────────────────
        case "Water leak":
            return "location" if ctx["catch_water_in_bucket"] == "yes" else "diy"

        case "Sealant damaged around bath/basin or sink":
            return "location" if ctx["black_mold"] == "yes" else "diy"

        case "Shower curtain rail broken" | "Mixer shower not working":
            return "location" if ctx["fitted_by_midland"] == "yes" else "diy"

        # ──────────────────────────── Toilets ─────────────────────────────
        case "My only toilet is blocked" | "One of my toilets is blocked":
            return "location" if ctx["unblock_yourself"] == "yes" else "diy"

        case (
            "Toilet is loose from the floor and I don't feel like I can use it"
            | "Cistern behind toilet loose / unstable"
            | "Broken pipe behind toilet"
        ):
            return "location" if ctx["only_toilet"] == "yes" else "diy"

        # ───────────────────────────── Doors ──────────────────────────────
        case "Fire door is sticking / loose / draughty":
            return "location" if ctx["pass_smoke"] == "yes" else "diy"

        case "Door frame split or damaged":
            return "location" if ctx["door_open_inside_corridor"] == "yes" else "diy"

        # ───────────────────────────── Locks ──────────────────────────────
        case "Door lock is sticking - my property is secure":
            return "location" if ctx["placeholder"] == "yes" else "diy"

        # ──────────────────────────── Windows ─────────────────────────────
        case "Smashed window":
            return "location" if ctx["crime_ref_num"] == "yes" else "location"

        # ─────────────────────────── Lighting / Power ─────────────────────
        case "No lights working in my property":
            return "location" if ctx["checked_trip_switch"] == "yes" else "diy"

        case "Some lights not working":
            return "location" if ctx["any_lights_working"] == "yes" else "location"

        case "No power in my property":
            return "location" if ctx["checked_trip_switch"] == "yes" else "diy"

        # ─────────────────────── Stair & Through-floor Lifts ──────────────
        case (
            "Stair Lift is not working / moving"
            | "Damage to stair lift (still working)"
            | "Through floor lift not working"
            | "Damage to through floor lift (still working)"
        ):
            return "location" if ctx["fitted_by_midland"] == "yes" else "diy"

        # ──────────────── Gas Heating & Hot-Water (existing) ──────────────
        case "Gas boiler low pressure - no water leak":
            return "location" if ctx["re_pressurise_boiler"] == "yes" else "diy"

        case "Gas central heating not working":
            return "location" if ctx["checked_gas_meter"] == "yes" else "diy"

        case "Gas fire or heater not working":
            return "location" if ctx["only_heating_form"] == "yes" else "location"

        case "Hot water not working":
            return "location" if ctx["checked_gas_meter"] == "yes" else "diy"

        # Newly-added gas / boiler sub-issues (placeholder logic)
        case (
            "Pipes have started making loud and unusual noises (I've not heard before)"
            | "Error code on boiler display"
            | "Gas fire or heater damaged"
            | "Gas boiler leaking water on electrical fittings"
            | "Gas boiler water leaking"
        ):
            return "location" if ctx["placeholder"] == "yes" else "diy"

        # ─────── Electric / Storage Heating, Radiators & Showers ──────────
        case (
            "No hot water from electric (immersion) heater"
            | "An electric heater or fire is not working"
            | "Hot Water Cylinder is Leaking"
            | "Electric heating not working"
            | "Air Source Heating - not working"
            | "Radiator loose - coming away from the wall"
            | "One of my radiators isn't getting warm"
            | "Radiator needs putting back on the wall"
            | "Radiator leaking - electrical fittings getting wet"
            | "Radiator leaking - flooding the property"
            | "Radiator leaking - collecting water in a bucket"
            | "Electrical shower unit leaking"
            | "Electric shower hose, handset or rail broken"
            | "Electric shower not working / no hot water"
        ):
            return "location" if ctx["placeholder"] == "yes" else "diy"

        # ───────────────────────────── Fences ─────────────────────────────
        case "Fence panels missing or damaged":
            return (
                "location"
                if ctx["fence_location"] == "Next to a public footpath or road"
                else "diy"
            )

        # ─────────────────────────── Groundworks ──────────────────────────
        case (
            "Driveway has multiple large cracks"
            | "Garden path/step is loose/broken"
            | "Personal rotary clothes line broken"
            | "Retractable clothes line & post (in any area) broken"
            | "Concrete post for personal clothes line broken"
        ):
            return "location" if ctx["placeholder"] == "yes" else "diy"

        # ──────────────────────────── Plumbing ────────────────────────────
        case (
            "Blocked drain overflowing sewage"
            | "Blocked drain"
            | "Drain cover broken (no trip hazard)"
            | "Drain cover missing (fall or trip hazard)"
            | "Outside tap leaking or loose"
            | "Outside tap dripping"
            | "Gutter or drainpipe blocked or broken"
        ):
            return "location" if ctx["placeholder"] == "yes" else "diy"

        # ──────────────────────────── Catch-all ───────────────────────────
        case _:
            # If we somehow miss an issue_type, default sensibly:
            return "diy"
