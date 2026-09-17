# P23-R R0 incident remediation

The prior mixed-role CSV inspection remains preserved in
`docs/v7_g2_p23_test_access_incident.json` and is not overwritten.  Its
`test_iid` rows are permanently denylisted for P23-R.

For P23-R, `test_iid` is treated only as historical/legacy opened evidence;
no new test labels, predictions, errors, or mixed-role diagnostics are read.
The sealed IID split and DeepOHeat official-100 evaluation remain untouched.
The incident did not alter model selection or downstream metrics.  This replay
uses an independently staged `valid_iid` manifest with 128 frozen case IDs and
truth-row indices, followed by inference from frozen checkpoints only.  No
training, checkpoint reselection, or test unlock is allowed.

The machine-readable receipt is
`docs/v7_g2_p23_r0_incident_remediation.json`.
