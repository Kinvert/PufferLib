"""Conservative history scaling for measured offline reporting time."""
import math

from image_campaign_budget import validate
from candidate_panel import require


def seconds(calibration, observations, optimizer_seconds=()):
    validate(calibration)
    require(type(observations) is int and observations >= 0,
            "Reporting reserve needs a nonnegative complete-panel count")
    pilot_count=len(calibration["measured_panel_seconds"])
    scale=max(1.,observations/pilot_count)
    require(isinstance(optimizer_seconds,(list,tuple))
            and all(type(t) in (int,float) and math.isfinite(t) and t>0 for t in optimizer_seconds),
            "Invalid completed optimizer timing observation")
    # Repeating the entire pilot report per pilot-sized block overcharges its
    # fixed Nature/source audit. It avoids treating a larger history as free.
    final=max(calibration["measured_final_replay_seconds"],max(optimizer_seconds,default=0.))
    result=calibration["safety_factor"]*(final
        +calibration["measured_report_seconds"]*scale)
    require(math.isfinite(result) and result>0,"Invalid history-scaled reporting reserve")
    return result
