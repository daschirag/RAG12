"""B1 - Hand-written source material for the SYNTHETIC knowledge-base documents.

Everything in this module was written for this project as general engineering
guidance in our own words. It is not copied from any manufacturer manual or
standard, contains no proprietary limits or clause numbers, and every document
built from it is marked synthetic. It must not be used for real operations.
"""

from __future__ import annotations

# Per wind-turbine component: purpose, typical symptoms, likely causes,
# troubleshooting checks and corrective actions.
WIND_COMPONENT_GUIDE: dict[str, dict[str, object]] = {
    "transformer": {
        "purpose": "Steps the generator/converter voltage up to the collection-grid voltage. Oil-filled or dry-type "
                   "units sit in the nacelle, tower base or a separate container and are cooled by air or oil circulation.",
        "symptoms": ["winding or core temperature rising faster than load explains",
                     "high temperature in the transformer cell or container",
                     "pressure-relief or overpressure alarm on an oil-filled unit",
                     "protection trip and loss of grid connection"],
        "causes": ["blocked or failed cell ventilation", "sustained operation near full load in hot weather",
                   "insulation ageing leading to internal partial discharge", "loose connections causing local heating",
                   "moisture ingress in oil-filled units"],
        "checks": ["Compare the three phase temperatures with each other and with the loading; one hot phase points "
                   "to a connection or winding issue, all phases hot points to cooling.",
                   "Check cell ventilation fans, filters and louvres for blockage.",
                   "Review the temperature trend for the past weeks for a gradual drift.",
                   "For oil-filled units, inspect oil level, leaks and the pressure-relief device; arrange an oil sample."],
        "actions": ["Restore ventilation (clean filters, repair fans) before returning to full load.",
                    "Derate the turbine while temperatures remain above the normal band.",
                    "Tighten or replace discoloured connections after isolation.",
                    "Escalate to a transformer specialist if oil analysis or winding tests are abnormal."],
        "safety": "Work on the transformer only after isolation from the medium-voltage network, earthing and a "
                  "permit-to-work; allow hot equipment to cool.",
    },
    "gearbox": {
        "purpose": "Increases the slow rotor speed to the generator speed through planetary and parallel stages, "
                   "lubricated and cooled by a pumped gear-oil circuit with filters and a radiator.",
        "symptoms": ["rising bearing or oil temperature at unchanged power", "oil level or oil pressure alarms",
                     "increasing filter pollution indicator", "new vibration components or noise",
                     "oil leaks around seals and pipework"],
        "causes": ["bearing or gear-tooth wear and pitting", "degraded or contaminated oil",
                   "failed oil pump, coupling or cooler bypass valve", "clogged filters", "seal failure"],
        "checks": ["Compare gearbox bearing temperatures across bearings and against sister turbines at similar power.",
                   "Check oil level, oil pressure and filter differential pressure readings.",
                   "Inspect the oil pump motor current and coupling; a pump running with no pressure rise suggests a "
                   "coupling failure.",
                   "Take an oil sample for particle count and wear-metal analysis.",
                   "Perform an endoscope inspection if temperatures or vibration keep rising."],
        "actions": ["Replace filters and correct oil level; repair leaks.", "Replace a failed pump coupling or valve.",
                    "Plan a bearing or gear-stage exchange when inspection confirms damage; derate until then.",
                    "Shorten the oil-sampling interval after any gearbox alarm."],
        "safety": "Lock the rotor before entering the drive train area; hot oil and pressurised lines can cause burns.",
    },
    "generator_bearing": {
        "purpose": "Supports the generator shaft at the drive end and non-drive end; greased or oil-lubricated rolling "
                   "bearings with temperature sensors.",
        "symptoms": ["bearing temperature rising relative to the other bearing and to ambient",
                     "temperature rise that persists at moderate power", "increased vibration at bearing frequencies",
                     "grease leakage or discoloured grease"],
        "causes": ["insufficient or degraded lubrication", "electrical current passing through the bearing (fluting)",
                   "misalignment between gearbox and generator", "contamination"],
        "checks": ["Compare drive-end and non-drive-end temperatures; a growing difference is a typical early sign.",
                   "Check the automatic lubrication system for empty reservoirs or blocked lines.",
                   "Inspect shaft grounding brushes and insulation.", "Check alignment records after any drive-train work."],
        "actions": ["Restore lubrication and monitor the trend.", "Replace worn grounding brushes.",
                    "Schedule a bearing exchange when temperature or vibration keeps rising; plan crane logistics early."],
        "safety": "Stop and lock the turbine before opening the generator; rotating parts and electrical hazards.",
    },
    "rotor_main_bearing": {
        "purpose": "Carries the rotor weight and thrust loads on the main shaft; failures usually force a long standstill.",
        "symptoms": ["rising main-bearing temperature", "noise or vibration from the main shaft",
                     "grease leakage at the seals", "turbine stopped by bearing protection"],
        "causes": ["lubrication failure", "high dynamic loads and misalignment", "contamination and seal damage",
                   "material fatigue at end of life"],
        "checks": ["Trend all main-bearing temperature sensors and compare with sister turbines.",
                   "Check grease supply and purge; inspect purged grease for metal particles.",
                   "Review vibration measurements on the main shaft.",
                   "Inspect seals and the rotor lock area."],
        "actions": ["Stop the turbine if temperatures keep rising after lubrication is restored.",
                    "Plan a main-bearing exchange with a large crane; this is a major repair.",
                    "Consider a temporary derating only with engineering approval."],
        "safety": "Main-bearing work requires the rotor to be locked and often removed; follow the lifting plan.",
    },
    "converter": {
        "purpose": "Power electronics that control generator torque and feed power to the grid; includes grid-side and "
                   "rotor/generator-side inverters, a DC link, filters, fuses and a liquid or air cooling system.",
        "symptoms": ["converter fault trip and restart attempts", "IGBT or cabinet temperature alarms",
                     "blown fuses in the filter or supply", "torque oscillation alarms", "reduced power output"],
        "causes": ["failed power modules or drivers", "fuse failure", "cooling problems in the converter cabinet",
                   "control-board or firmware faults", "grid disturbances"],
        "checks": ["Read the converter fault log for the exact fault code and time.",
                   "Check IGBT and cabinet temperatures and the converter cooling circuit.",
                   "Inspect filter fuses and contactors after isolation.",
                   "Check for simultaneous grid events at other turbines."],
        "actions": ["Replace failed fuses or modules; verify the cause before restarting.",
                    "Repair converter cooling (pumps, fans, coolant level).",
                    "Update or reload converter firmware if the manufacturer advises it."],
        "safety": "DC-link capacitors can hold a dangerous charge after isolation; wait and verify zero voltage.",
    },
    "hydraulic_system": {
        "purpose": "Supplies pressurised oil to brakes, the rotor lock and, on some turbines, the pitch or yaw system; "
                   "consists of a pump group, tank, accumulators, valves and filters.",
        "symptoms": ["pressure not building or falling too quickly", "pump running more often than normal",
                     "hydraulic oil temperature rising", "oil leaks", "accumulator pre-charge alarms"],
        "causes": ["internal leakage in valves or cylinders", "worn pump", "accumulator losing its gas pre-charge",
                   "clogged filters", "external leaks"],
        "checks": ["Compare pump running time and pressure build-up time with previous weeks.",
                   "Check accumulator pre-charge pressure.", "Inspect for leaks and check the oil level.",
                   "Check the hydraulic oil temperature and tank heating."],
        "actions": ["Recharge or replace accumulators.", "Replace leaking seals, hoses or valves.",
                    "Replace the pump when the build-up time keeps increasing."],
        "safety": "Release stored pressure before opening any hydraulic line; accumulators stay charged after shutdown.",
    },
    "pitch_system": {
        "purpose": "Turns each blade about its axis to control power and to stop the turbine; each axis has a motor or "
                   "cylinder, a controller, position encoders and a backup energy store (batteries or accumulators).",
        "symptoms": ["axis not ready or pitch fault stops", "blade position disagreement between axes",
                     "low DC-link or battery voltage alarms", "pitch motor or cabinet over-temperature",
                     "communication faults in the hub"],
        "causes": ["aged or failing backup batteries or chargers", "defective encoders", "failed pitch controller cards",
                   "slip-ring wear interrupting power or signals to the hub", "motor or fan failures"],
        "checks": ["Identify which axis reports the fault and compare its signals with the other two.",
                   "Check battery voltages and the last battery test result.",
                   "Check encoder signals for jumps or dropouts.", "Inspect the slip ring and hub wiring connectors.",
                   "Check motor temperature and cooling fan operation."],
        "actions": ["Replace batteries or chargers that fail the capacity test.", "Replace defective encoders or cards.",
                    "Clean or replace the slip ring; repair damaged hub wiring.", "Replace failed pitch motor fans."],
        "safety": "Enter the hub only with the rotor locked; the pitch backup energy can move blades unexpectedly.",
    },
    "rotor_brake": {
        "purpose": "Mechanical disc brake used to stop and hold the rotor for maintenance, usually hydraulically operated.",
        "symptoms": ["brake cannot be applied or released", "brake pad wear alarms",
                     "hydraulic pressure at the brake not reached", "supply faults for the brake control circuit"],
        "causes": ["hydraulic supply problems", "worn pads or a damaged disc", "faulty brake valves",
                   "control supply failures"],
        "checks": ["Check brake circuit pressure and valve operation.", "Inspect pad thickness and the disc surface.",
                   "Check the brake control supply voltage."],
        "actions": ["Replace pads or valves as needed.", "Repair the hydraulic supply.",
                    "Restore the control supply and test the brake before returning to service."],
        "safety": "Never rely on a brake with an open fault; use the rotor lock for work in the drive train.",
    },
    "yaw_system": {
        "purpose": "Turns the nacelle into the wind using yaw motors and gears, holding it with yaw brakes; greased "
                   "gear teeth and bearing.",
        "symptoms": ["yaw misalignment alarms", "yaw motor over-current", "noise during yawing",
                     "grease pump faults", "cable twist alarms"],
        "causes": ["failed yaw motors or brakes", "lubrication failure", "worn carbon brushes or damaged gear teeth"],
        "checks": ["Compare yaw motor currents.", "Check the automatic greasing system and grease reservoir.",
                   "Inspect yaw brake pads and the yaw ring teeth."],
        "actions": ["Replace defective motors or brushes.", "Repair the grease pump and refill.",
                    "Replace worn yaw brake pads."],
        "safety": "Isolate yaw drives before working near the yaw ring.",
    },
    "cooling_system": {
        "purpose": "Removes heat from the generator, converter, gearbox and cabinets with water-glycol and/or air "
                   "circuits, pumps, fans, valves and heat exchangers.",
        "symptoms": ["component temperatures rising together", "low flow or low pressure in the cooling circuit",
                     "fan or pump motor current abnormal", "coolant leaks"],
        "causes": ["valves left in the wrong position after maintenance", "failed pumps or fans", "air in the circuit",
                   "blocked heat exchangers", "coolant loss"],
        "checks": ["Check valve positions against the as-built drawing, especially after maintenance.",
                   "Check pump and fan currents and circuit pressure and flow.", "Inspect for leaks and coolant level."],
        "actions": ["Correct valve positions and bleed air.", "Replace failed pumps or fans.",
                    "Clean heat exchangers; top up or replace coolant."],
        "safety": "Hot pressurised coolant: depressurise and cool down before opening the circuit.",
    },
    "electrical_auxiliary": {
        "purpose": "Low-voltage supplies for control, lighting, heaters and the safety chain (24 V supplies, residual "
                   "current devices, internal-consumption metering).",
        "symptoms": ["safety chain open", "24 V supply faults", "RCD trips", "implausible internal-consumption readings"],
        "causes": ["failed power supplies", "earth faults", "defective relays or measurement transducers",
                   "loose terminals"],
        "checks": ["Locate the open element of the safety chain from the controller log.",
                   "Measure supply voltages under load.", "Test RCDs and insulation resistance."],
        "actions": ["Replace failed supplies, relays or transducers.", "Repair earth faults before resetting RCDs."],
        "safety": "Follow electrical isolation procedures; never bridge the safety chain to restart a turbine.",
    },
    "communication_control": {
        "purpose": "Turbine controller, fieldbus couplers and network links between nacelle, hub and tower base.",
        "symptoms": ["communication fault alarms", "missing or frozen SCADA values",
                     "turbine stops because a subsystem is not responding"],
        "causes": ["failed bus couplers or network modules", "damaged plugs or cables (especially at the hub interface)",
                   "power supply dips", "firmware faults"],
        "checks": ["Identify which node stopped communicating and when.", "Inspect connectors and cables at that node.",
                   "Check the supply voltage of the node."],
        "actions": ["Replace defective couplers or plugs; repair wiring.", "Restore the configuration and firmware."],
        "safety": "Some checks need access to the hub or nacelle; apply the relevant lock-out procedure.",
    },
    "blades_hub_structure": {
        "purpose": "Blades, spinner and hub structure including lightning protection, grounding rollers, grease "
                   "collectors, covers and caps.",
        "symptoms": ["findings during periodic inspection (missing caps, grease collectors, damaged covers)",
                     "lightning protection or grounding defects", "spinner component damage"],
        "causes": ["wear, vibration loosening", "lightning strikes", "incomplete re-assembly after service"],
        "checks": ["Inspect against the periodic inspection checklist.", "Measure lightning-protection continuity.",
                   "Check that all covers and collectors are fitted."],
        "actions": ["Refit or replace missing parts.", "Repair lightning-protection defects before the next storm season."],
        "safety": "Blade and hub work needs rope access or a platform and a locked rotor.",
    },
}

# Plan's solar-inverter fault list (synthetic knowledge; the dataset has no inverter fault labels).
SOLAR_INVERTER_GUIDE: dict[str, dict[str, object]] = {
    "inverter_overheating": {
        "title": "Inverter overheating",
        "symptoms": ["heat-sink or cabinet temperature alarms", "output reduced by thermal derating at midday",
                     "inverter trips on hot afternoons"],
        "causes": ["blocked air filters or failed fans", "high ambient temperature and poor enclosure ventilation",
                   "ageing power modules"],
        "checks": ["Compare AC output with irradiation: derating shows as a flat top at midday.",
                   "Check fans and filters.", "Compare with neighbouring inverters under the same irradiation."],
        "actions": ["Clean filters and replace fans.", "Improve enclosure ventilation or shading.",
                    "Replace modules that keep overheating."],
    },
    "abnormal_voltage": {
        "title": "Abnormal DC or AC voltage",
        "symptoms": ["DC input voltage outside the expected band for the irradiation and module temperature",
                     "grid over/under-voltage trips"],
        "causes": ["string faults (open circuits, damaged connectors)", "grid disturbances", "incorrect string sizing"],
        "checks": ["Compare the DC voltage with other inverters.", "Measure string open-circuit voltages.",
                   "Check the grid voltage at the point of connection."],
        "actions": ["Repair or replace damaged strings or connectors.", "Coordinate with the grid operator on grid issues."],
    },
    "abnormal_current": {
        "title": "Abnormal DC or AC current",
        "symptoms": ["string or inverter current lower than peers under the same irradiation",
                     "current imbalance between phases"],
        "causes": ["soiling or shading of part of the array", "failed bypass diodes or modules", "blown string fuses"],
        "checks": ["Compare per-string currents.", "Inspect the array for soiling and shading.",
                   "Check fuses and combiner boxes."],
        "actions": ["Clean or remove the shading source.", "Replace failed fuses, diodes or modules."],
    },
    "underperformance": {
        "title": "Underperformance",
        "symptoms": ["performance ratio (energy / irradiation) lower than the plant's usual level",
                     "one inverter producing less than its neighbours for days"],
        "causes": ["soiling", "degradation", "partial string outages", "inverter derating", "incorrect sensor readings"],
        "checks": ["Trend the ratio of AC energy to irradiation per inverter.",
                   "Rule out a faulty irradiation sensor by comparing with a second sensor or nearby plant.",
                   "Look for stepwise drops (outage) versus gradual drift (soiling, degradation)."],
        "actions": ["Schedule cleaning.", "Repair failed strings.", "Recalibrate or replace sensors."],
    },
    "sensor_anomaly": {
        "title": "Sensor anomaly",
        "symptoms": ["irradiation or temperature readings that disagree with power output",
                     "flat-lined or jumping sensor values", "night-time irradiation readings above zero"],
        "causes": ["sensor soiling or misalignment", "cabling faults", "data logger problems", "calibration drift"],
        "checks": ["Compare the sensor with power output and with other sensors.",
                   "Check for constant values or sudden steps.", "Inspect sensor mounting and cleanliness."],
        "actions": ["Clean, realign or recalibrate the sensor.", "Repair cabling or the logger.",
                    "Flag affected data so it is not used for performance analysis."],
    },
    "power_generation_deviation": {
        "title": "Power-generation deviation",
        "symptoms": ["measured output persistently away from the forecast or from expected output for the weather",
                     "sudden drops to zero while irradiation is high"],
        "causes": ["inverter trips", "curtailment", "string outages", "forecast error rather than a plant fault"],
        "checks": ["Check whether the deviation coincides with an inverter trip or curtailment order.",
                   "Compare with the forecast prediction interval: values inside the interval are not a fault signal.",
                   "Check neighbouring inverters."],
        "actions": ["Restart or repair tripped inverters.", "Record curtailment separately from faults.",
                    "Investigate strings if the deviation persists."],
    },
}

# Short original scope overviews. They describe what each standard is about in general
# terms only; they do not reproduce or paraphrase specific clauses or requirements.
STANDARD_OVERVIEWS: dict[str, dict[str, str]] = {
    "iec_61400_1": {
        "name": "IEC 61400-1",
        "topic": "Design requirements for wind turbines",
        "overview": "Part of the IEC 61400 series on wind energy generation systems. It addresses the design of wind "
                    "turbines so that they withstand the loads and environmental conditions of their site class over "
                    "their design life. For this project it is background for why drive-train and structural faults "
                    "are treated as high consequence.",
        "domain": "wind",
    },
    "iec_61400_12_1": {
        "name": "IEC 61400-12-1",
        "topic": "Power performance measurement of wind turbines",
        "overview": "Describes how the power curve of a wind turbine is measured and reported. The theoretical power "
                    "curve column in the wind dataset plays a similar role: comparing measured power with an expected "
                    "curve is how underperformance and downtime are recognised.",
        "domain": "wind",
    },
    "iec_61400_25": {
        "name": "IEC 61400-25",
        "topic": "Communications for monitoring and control of wind power plants",
        "overview": "Defines an information model and communication services so that SCADA systems can exchange "
                    "turbine data and control commands in a uniform way. It is relevant to how SCADA signals such as "
                    "the 10-minute averages in the fault dataset are structured and exchanged.",
        "domain": "wind",
    },
    "iec_61724_1": {
        "name": "IEC 61724-1",
        "topic": "Photovoltaic system performance monitoring",
        "overview": "Covers the monitoring of PV plant performance: which quantities to measure (for example "
                    "irradiance, temperatures and electrical output), and how to compute performance indicators. It is "
                    "background for the underperformance and sensor-anomaly knowledge in this knowledge base.",
        "domain": "solar",
    },
    "iec_62446_1": {
        "name": "IEC 62446-1",
        "topic": "PV systems: testing, documentation and maintenance",
        "overview": "Addresses the documentation, commissioning tests and inspection of grid-connected PV systems. It "
                    "is background for the solar troubleshooting procedures (string checks, inspections).",
        "domain": "solar",
    },
    "iso_17359": {
        "name": "ISO 17359",
        "topic": "Condition monitoring and diagnostics of machines - general guidelines",
        "overview": "Gives a general approach to setting up condition monitoring: choosing what to measure, setting "
                    "alert criteria, diagnosing and deciding on action. The fault-diagnosis track of this project "
                    "follows the same monitor-detect-diagnose-act pattern.",
        "domain": "general",
    },
    "iso_55000": {
        "name": "ISO 55000",
        "topic": "Asset management - overview and principles",
        "overview": "Introduces asset management as the coordinated activity of an organisation to realise value from "
                    "its assets, balancing cost, risk and performance. It frames why maintenance decisions in this "
                    "project are risk-based and why the trust-aware decision engine asks for evidence.",
        "domain": "general",
    },
}

# Invented maintenance findings/actions per component, used to write synthetic work orders
# attached to real event windows. Findings are NOT known facts about those events.
MAINTENANCE_TEMPLATES: dict[str, tuple[str, str]] = {
    "transformer": ("Cell ventilation filters found heavily clogged; phase temperatures uneven.",
                    "Filters replaced, one fan motor exchanged, connections checked. Returned to service with a "
                    "temperature watch for two weeks."),
    "gearbox": ("Oil sample showed elevated wear particles; filter pollution indicator high.",
                "Filters replaced, oil topped up, endoscope inspection scheduled. Oil-sampling interval shortened."),
    "generator_bearing": ("Drive-end bearing temperature above non-drive end; grease reservoir nearly empty.",
                          "Lubrication system refilled and purged, grounding brushes replaced. Bearing exchange planned "
                          "if the temperature gap persists."),
    "rotor_main_bearing": ("Main-bearing temperature trend rising; metal particles in purged grease.",
                           "Turbine kept stopped; main-bearing exchange with crane planned."),
    "converter": ("Converter fault log showed repeated trips; one filter fuse blown.",
                  "Fuse replaced after insulation check; converter cooling circuit verified. Restarted and monitored."),
    "hydraulic_system": ("Pump running time doubled compared with last month; accumulator pre-charge low.",
                         "Accumulator recharged, leaking valve seal replaced, oil level corrected."),
    "pitch_system": ("One pitch axis reported battery voltage below the test threshold.",
                     "Battery set on the affected axis replaced, charger tested, slip ring cleaned."),
    "rotor_brake": ("Brake circuit pressure not reached on command.",
                    "Brake valve replaced and brake function tested before handover."),
    "yaw_system": ("Yaw grease pump not delivering; reservoir empty.",
                   "Grease pump replaced and reservoir refilled; yaw ring lubrication checked."),
    "cooling_system": ("A cooling-circuit valve was found in the wrong position.",
                       "Valve set to the correct position, circuit bled, temperatures confirmed normal."),
    "electrical_auxiliary": ("Safety chain opened by a failed relay; 24 V supply voltage low under load.",
                             "Relay and power supply replaced; RCD tested."),
    "communication_control": ("Bus coupler in the nacelle cabinet not responding; connector damaged.",
                              "Coupler and connector replaced; configuration restored and communication verified."),
    "blades_hub_structure": ("Periodic inspection found missing covers and a missing grease collector.",
                             "Missing parts refitted; lightning-protection continuity measured."),
}
