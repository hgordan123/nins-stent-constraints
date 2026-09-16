import numpy as np
import slicer

# Landmark_Measurements.py
# Script to parse a list of points and return mathematical values based upon those points.
# Node names are entered interactively, and each measurement can be skipped.
# Workflow:
#   1. Enter the 3 lines and compute SPG centroid
#   2. Pause to draw remaining curves/landmarks based on the centroid
#   3. Enter landmark list and run all measurements
#   4. Print a full summary of results at the end

############ FUNCTIONS #############################

def get_line_points(lineNode):
    """Returns two endpoints of a line markup in world coordinates."""
    p0 = [0.0, 0.0, 0.0]
    p1 = [0.0, 0.0, 0.0]
    lineNode.GetNthControlPointPositionWorld(0, p0)
    lineNode.GetNthControlPointPositionWorld(1, p1)
    return np.array(p0), np.array(p1)


def get_curve(name):
    """Loads a curve node by name and returns its interpolated points as an (N,3) array."""
    curve_node = slicer.util.getNode(name)
    interpolated_curve = curve_node.GetCurvePointsWorld()
    curve_points = np.array([
        interpolated_curve.GetPoint(i)
        for i in range(interpolated_curve.GetNumberOfPoints())
    ])
    return curve_points


def in_plane_distance(p1, p2, axis):
    """Returns Euclidean distance between p1 and p2 ignoring the specified axis (0=X, 1=Y, 2=Z)."""
    axes_to_use = [i for i in range(3) if i != axis]
    diff = np.array(p2)[axes_to_use] - np.array(p1)[axes_to_use]
    return np.linalg.norm(diff)


def find_closest_point(point, curve_name, axis_xyz):
    """
    Finds the closest point on a named curve to a given point.
    Returns the closest point, its index, and the in-plane distance (ignoring axis_xyz).
    Also places a fiducial markup at the closest point for visualization.
    """
    curve_points = get_curve(curve_name)
    distances = np.linalg.norm(curve_points - point, axis=1)
    closest_point_index = np.argmin(distances)
    closest_point = curve_points[closest_point_index]

    closest_point_node = slicer.vtkMRMLMarkupsFiducialNode()
    slicer.mrmlScene.AddNode(closest_point_node)
    closest_point_node.AddControlPoint(closest_point)
    closest_point_node.SetName('Closest Point on Curve to ' + curve_name)

    distance = in_plane_distance(point, closest_point, axis_xyz)
    return closest_point, closest_point_index, distance


def closest_distance_between_curves(curve1_name, curve2_name):
    """Returns the closest points on two curves and the minimum distance between them."""
    curve1 = get_curve(curve1_name)
    curve2 = get_curve(curve2_name)

    diff = curve1[:, np.newaxis, :] - curve2[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)

    min_index = np.unravel_index(np.argmin(distances), distances.shape)
    i, j = min_index

    return curve1[i], curve2[j], distances[i, j]


def best_intersection_point(PA, PB):
    """
    Given arrays PA, PB (each Nx3) as endpoints of N lines,
    returns the point that minimizes the sum of squared distances to each line.
    Uses each PA[i] as the known point on the line for accuracy.
    """
    A_stack = np.zeros((0, 3))
    b_stack = np.zeros((0,))

    for i in range(len(PA)):
        direction = PB[i] - PA[i]
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            print(f'  Warning: line {i} has near-zero length, skipping.')
            continue
        d = direction / norm
        M = np.eye(3) - np.outer(d, d)
        A_stack = np.vstack([A_stack, M])
        b_stack = np.concatenate([b_stack, M @ PA[i]])

    P, residuals, rank, s = np.linalg.lstsq(A_stack, b_stack, rcond=None)
    return P, residuals


def prompt_node_name(description, required=True):
    """
    Prompts the user to enter a node name.
    Returns the entered name, or None if skipped.
    Does not loop to avoid EOFError in Slicer exec() context.
    """
    if required:
        name = input(f'  Enter node name for [{description}]: ').strip()
        return name if name else None
    else:
        name = input(f'  Enter node name for [{description}] (or press Enter to skip): ').strip()
        return name if name else None


def ask_skip(measurement_name):
    """Asks the user whether to run a measurement. Returns True if they want to skip.
    Type 'q' at any prompt to quit the script and print results so far.
    Defaults to running (y) if input is not recognised."""
    ans = input(f'\nRun measurement [{measurement_name}]? (y/n/q to quit): ').strip().lower()
    if ans in ('q', 'quit'):
        print('\n  Quitting early — printing results collected so far.')
        print('\n' + '=' * 60)
        print(' RESULTS SUMMARY (partial)')
        print('=' * 60)
        for label, value in results.items():
            print(f'  {label}: {value}')
        print('=' * 60)
        raise SystemExit
    if ans in ('n', 'no'):
        return True
    return False


def safe_get_node(name, description):
    """Tries to get a node by name. Prints a clear error and returns None if not found."""
    if not name:
        return None
    try:
        return slicer.util.getNode(name)
    except Exception:
        print(f'  ERROR: Could not find node "{name}" for [{description}]. Skipping.')
        return None


def prompt_and_get_node(description):
    """
    Prompts the user to enter a curve/node name.
    Gives one retry if the node is not found.
    Press Enter with no input to skip immediately.
    """
    name = input(f'  Enter node name for [{description}] (or press Enter to skip): ').strip()
    if not name or name == '':
        return None, None
    try:
        node = slicer.util.getNode(name)
        if node is None:
            raise Exception('Node is None')
        return node, name
    except Exception:
        print(f'  Node "{name}" not found. Check capitalisation and spelling.')
        name = input(f'  Try again (or press Enter to skip): ').strip()
        if not name or name == '':
            return None, None
        try:
            node = slicer.util.getNode(name)
            if node is None:
                raise Exception('Node is None')
            return node, name
        except Exception:
            print(f'  Node "{name}" still not found — skipping this measurement.')
            return None, None


def prompt_landmark_index(landmark_name, num_points):
    """
    Prompts the user for the position (0-based index) of a named landmark.
    Returns the integer index, or None if skipped or invalid.
    Does not loop to avoid EOFError in Slicer exec() context.
    """
    raw = input(f'  What is the position of [{landmark_name}] in the landmarks list? '
                f'(0-{num_points - 1}, or press Enter to skip): ').strip()
    if raw == '':
        return None
    try:
        ind = int(raw)
        if 0 <= ind < num_points:
            return ind
        else:
            print(f'  Out of range (0-{num_points - 1}) — skipping.')
            return None
    except ValueError:
        print('  Invalid input — skipping.')
        return None


############ MAIN #############################

# Dictionary to collect all results for the final summary
results = {}

print('=' * 60)
print(' Landmark Measurements Script')
print('=' * 60)


# ── STEP 1: Compute SPG centroid from 3 lines ────────────────────
print('\n--- STEP 1: SPG Centroid (3-line intersection) ---')
print('  Before continuing, ensure you have placed 3 lines in the scene:')
print('  LINE 1 (Axial): Through the PPF and vidian canal, along the lateral border. e.g. vidian_ax')
print('  LINE 2 (Axial): Along the lateral border of the PPF, intersecting Line 1. e.g. PPF_ax')
print('  LINE 3 (Sagittal): Along the GP canal where it intersects the vidian canal. e.g. GP_sag')
print('  Now enter the node names for each line:')

SPG = None

def prompt_line_node(description):
    """Prompt for a line node name, with one retry if not found."""
    name = prompt_node_name(description)
    if not name or name == '':
        return None, None
    ln = safe_get_node(name, description)
    if ln is not None:
        return ln, name
    # One retry
    print(f'  Node "{name}" not found. Please check the name and try once more.')
    name2 = prompt_node_name(description)
    if not name2 or name2 == '':
        return None, None
    ln2 = safe_get_node(name2, description)
    if ln2 is None:
        print(f'  Node "{name2}" still not found. Please re-run the script.')
        raise SystemExit
    return ln2, name2

PA, PB = [], []

ln1, name1 = prompt_line_node('Vidian canal axial line (e.g. vidian_ax)')
if ln1 is None:
    print('  ERROR: Vidian canal line not found. Please re-run the script.')
    raise SystemExit
p0, p1 = get_line_points(ln1)
PA.append(p0); PB.append(p1)
print(f'  {name1}: {p0} -> {p1}')

ln2, name2 = prompt_line_node('PPF axial line (e.g. PPF_ax)')
if ln2 is None:
    print('  ERROR: PPF axial line not found. Please re-run the script.')
    raise SystemExit
p0, p1 = get_line_points(ln2)
PA.append(p0); PB.append(p1)
print(f'  {name2}: {p0} -> {p1}')

ln3, name3 = prompt_line_node('GP canal sagittal line (e.g. GP_sag)')
if ln3 is None:
    print('  ERROR: GP canal sagittal line not found. Please re-run the script.')
    raise SystemExit
p0, p1 = get_line_points(ln3)
PA.append(p0); PB.append(p1)
print(f'  {name3}: {p0} -> {p1}')

PA = np.array(PA)
PB = np.array(PB)
SPG, residuals = best_intersection_point(PA, PB)
print(f'\n  SPG best intersection: {SPG}')
print(f'  Residuals: {residuals}')

SPG_centroid_node = slicer.vtkMRMLMarkupsFiducialNode()
slicer.mrmlScene.AddNode(SPG_centroid_node)
SPG_centroid_node.AddControlPoint(SPG.tolist())
SPG_centroid_node.SetName('SPG_Centroid')
print('  SPG_Centroid markup placed in scene.')



# ── STEP 2: Guided landmark and curve placement ─────────────────
print('\n' + '=' * 60)
print(' STEP 2: PLACE LANDMARKS AND CURVES')
print(f' SPG Centroid coordinates: {SPG}')
print('=' * 60)
print("""
  The SPG centroid has been placed in the scene. Use its location
  to guide all landmark and curve placement below.

  ── A. SPF DIAMETER LINE MEASUREMENTS ───────────────────────────
  Using the Line markup tool, draw:

    A1. SPF diameter — AXIAL plane: Draw a line across the full
        diameter of the SPF on the axial slice at the SPG centroid.
        Name: e.g. 'SPF_diam_ax'
    A2. SPF diameter — CORONAL plane: Draw a line across the full
        diameter of the SPF on the coronal slice at the level of
        the SPG centroid.
        Name: e.g. 'SPF_diam_cor'

  ── B. POINT LANDMARKS (add to your landmarks list) ─────────────
  Using the Markups Fiducial tool, add:

    B1. 'ANS'             — Anterior Nasal Spine, placed at the
                            midline on an axial slice.
    B2. 'SPF_inf_border'  — Most inferior border of the SPF,
                            placed on a coronal slice.
    B3. 'Nasal_floor_inf' — Most inferior midline border of the
                            nasal floor, placed on a coronal slice.
    B4. 'FR_inf_border'   — Most inferior border of the foramen
                            rotundum. On a coronal slice, identify
                            the sphenoid pentagon and mark the
                            inferior border of the foramen rotundum.
    B5. 'IOF_medial_border' — Point landmark at the MEDIAL border
                            of the narrowest part of the inferior
                            orbital fissure on a coronal slice at
                            or nearest to the SPF diameter.

  ── C. CURVES — AXIAL PLANE (at the level of the SPG) ──────────
  Switch to the axial slice at the SPG centroid Z-level.
  Using the Markups Curve tool, draw:

    C1. Medial border of the middle turbinate — curve along the
        medial surface of the middle turbinate.
        Name: e.g. 'MT_medial_ax'
    C2. Medial nasal septum — curve along the medial surface of
        the nasal septum on the same slice.
        Name: e.g. 'septum_ax'
        NOTE: Ensure C1 and C2 are NOT confluent — we will measure
        the shortest distance between them.
    C3. Nasal cavity outline — curve along the nasal cavity wall
        at this level. This may be the same as C1 (middle turbinate
        medial border) if appropriate.
        Name: e.g. 'nasal_cavity_ax'
    C4. Cheek — curve along the outer cheek wall at this axial level.
        Name: e.g. 'cheek_ax'

  ── D. CURVES — CORONAL PLANE (at the SPF inferior border level) ─
  Navigate to the coronal slice at the level of landmark B2
  (SPF inferior border).

    D1. Middle turbinate medial border — curve along the medial
        surface of the middle turbinate at this coronal level.
        Name: e.g. 'MT_medial_cor'
    D2. Inferior turbinate superior surface — curve along the
        superior surface of the inferior turbinate at this coronal level.
        Name: e.g. 'inf_meatus_sup_cor'
    D3. Inferior middle turbinate surface — curve along the most
        inferior surface of the middle turbinate at this coronal level.
        Name: e.g. 'MT_inf_cor'
    D4. Cheek — curve along the outer cheek wall at this coronal level.
        Name: e.g. 'cheek_cor'

  ── E. CURVES — CORONAL PLANE (at the level of the PPF) ─────────
  Navigate to the coronal slice at the level of the PPF.

    E1. Inferior orbital floor — curve along the most inferior
        border of the orbital floor at this coronal level.
        Name: e.g. 'orbital_floor_cor'

  ── F. LINE MEASUREMENTS (collected by script) ──────────────────
  F1. Lateral nasal wall thickness @ SPF — CORONAL plane:
    - Navigate to the coronal slice at the level of the SPF.
    - Using the Line markup tool, draw a line across the lateral
      nasal wall at this level (equivalent to the PPF distance).
    - Name it exactly: e.g. 'lat_nasal_wall_cor'
    - The script will prompt you to enter this node name.

  F2. Hard palate length — SAGITTAL plane:
    - Navigate to the sagittal slice at the level of the ANS
      (landmark B1).
    - Using the Line markup tool, draw a line along the hard palate
      at this level.
    - Name it exactly: e.g. 'hard_palate_sag'
    - The script will prompt you to enter this node name.
""")
print('  When all landmarks and curves are placed, proceed to enter node names below.')


# ── STEP 3: Load landmark list ───────────────────────────────────
print('--- STEP 3: Landmark List ---')
landmark_node_name = prompt_node_name('Landmark list (fiducial node)')
points_landmark_node = None
if landmark_node_name is not None:
    points_landmark_node = safe_get_node(landmark_node_name, 'Landmark list')
if points_landmark_node is None:
    print('  Landmark list not found. Please try again.')
    landmark_node_name = prompt_node_name('Landmark list (fiducial node)')
    if landmark_node_name is not None:
        points_landmark_node = safe_get_node(landmark_node_name, 'Landmark list')
    if points_landmark_node is None:
        print('  Landmark list still not found — landmark-based measurements will be skipped.')

ans_coordinates    = None
nasal_floor_coords = None
spf_inf_coords     = None
iof_coords         = None
fr_coords          = None
num_points         = 0

if points_landmark_node:
    num_points = points_landmark_node.GetNumberOfControlPoints()
    print(f'  Found {num_points} landmarks (positions 0 to {num_points - 1}).')

    print('  Landmarks in list:')
    for i in range(num_points):
        label   = points_landmark_node.GetNthControlPointLabel(i)
        desc    = points_landmark_node.GetNthControlPointDescription(i)
        display = label if label else desc if desc else '(no label)'
        print(f'    {i}: {display}')
    print()

    # Prompt for all landmark positions up front
    print('  Please enter the landmark list positions for each point below.')
    print('  Press Enter to skip any landmark you have not placed.\n')

    # Anterior Nasal Spine
    ind = prompt_landmark_index('Anterior Nasal Spine (ANS)', num_points)
    if ind is not None:
        ans_coordinates = np.array(points_landmark_node.GetNthControlPointPosition(ind))
        desc = points_landmark_node.GetNthControlPointDescription(ind)
        print(f'  ANS loaded from index {ind} ({desc}): {ans_coordinates}')
    else:
        print('  ANS skipped.')

    # SPF Inferior Border
    ind = prompt_landmark_index('SPF Inferior Border (SPF_inf_border)', num_points)
    if ind is not None:
        spf_inf_coords = np.array(points_landmark_node.GetNthControlPointPosition(ind))
        desc = points_landmark_node.GetNthControlPointDescription(ind)
        print(f'  SPF Inferior Border loaded from index {ind} ({desc}): {spf_inf_coords}')
    else:
        print('  SPF Inferior Border skipped.')

    # Nasal Floor
    ind = prompt_landmark_index('Nasal Floor (Nasal_floor_inf)', num_points)
    if ind is not None:
        nasal_floor_coords = np.array(points_landmark_node.GetNthControlPointPosition(ind))
        desc = points_landmark_node.GetNthControlPointDescription(ind)
        print(f'  Nasal Floor loaded from index {ind} ({desc}): {nasal_floor_coords}')
    else:
        print('  Nasal Floor skipped.')

    # Foramen Rotundum Inferior Border
    ind = prompt_landmark_index('Foramen Rotundum inferior border (FR_inf_border)', num_points)
    if ind is not None:
        fr_coords = np.array(points_landmark_node.GetNthControlPointPosition(ind))
        desc = points_landmark_node.GetNthControlPointDescription(ind)
        print(f'  Foramen Rotundum inferior border loaded from index {ind} ({desc}): {fr_coords}')
    else:
        print('  Foramen Rotundum skipped.')

    # Inferior Orbital Fissure medial border
    ind = prompt_landmark_index('IOF medial border of narrowest point (IOF_medial_border)', num_points)
    if ind is not None:
        iof_coords = np.array(points_landmark_node.GetNthControlPointPosition(ind))
        desc = points_landmark_node.GetNthControlPointDescription(ind)
        print(f'  IOF medial border loaded from index {ind} ({desc}): {iof_coords}')
    else:
        print('  IOF medial border skipped.')



# ── STEP 4: Measurements ─────────────────────────────────────────
print('\n--- STEP 4: Measurements ---')

if SPG is None:
    print('SPG centroid not available. Cannot run measurements.')
else:

    # ── SPF DIAMETER - AXIAL ─────────────────────────────────────
    if not ask_skip('SPF DIAMETER - AXIAL'):
        node, spf_ax_line = prompt_and_get_node('SPF diameter line - Axial plane (e.g. SPF_diam_ax)')
        if node is not None:
            p0, p1 = get_line_points(node)
            spf_diam_axial = np.linalg.norm(p1 - p0)
            print(f'  SPF Diameter (Axial): {spf_diam_axial:.2f} mm')
            results['SPF DIAMETER - AXIAL (mm)'] = f'{spf_diam_axial:.2f}'
        else:
            results['SPF DIAMETER - AXIAL (mm)'] = 'Skipped'
    else:
        results['SPF DIAMETER - AXIAL (mm)'] = 'Skipped'

    # ── SPF DIAMETER - CORONAL ───────────────────────────────────
    if not ask_skip('SPF DIAMETER - CORONAL'):
        node, spf_cor_line = prompt_and_get_node('SPF diameter line - Coronal plane (e.g. SPF_diam_cor)')
        if node is not None:
            p0, p1 = get_line_points(node)
            spf_diam_coronal = np.linalg.norm(p1 - p0)
            print(f'  SPF Diameter (Coronal): {spf_diam_coronal:.2f} mm')
            results['SPF DIAMETER - CORONAL (mm)'] = f'{spf_diam_coronal:.2f}'
        else:
            results['SPF DIAMETER - CORONAL (mm)'] = 'Skipped'
    else:
        results['SPF DIAMETER - CORONAL (mm)'] = 'Skipped'

    # ── ANS to SPF - SAGITTAL ONLY ───────────────────────────────
    if not ask_skip('ANS to SPF - SAGITTAL ONLY'):
        if ans_coordinates is not None and spf_inf_coords is not None:
            ans_to_spf_inplane = in_plane_distance(ans_coordinates, spf_inf_coords, axis=0)
            print(f'  ANS to SPF (sagittal, Y+Z only): {ans_to_spf_inplane:.2f} mm')
            results['ANS to SPF - SAGITTAL ONLY (mm)'] = f'{ans_to_spf_inplane:.2f}'
        else:
            print('  ANS or SPF Inferior Border not loaded (check Step 3).')
            results['ANS to SPF - SAGITTAL ONLY (mm)'] = 'Skipped (ANS or SPF not loaded)'
    else:
        results['ANS to SPF - SAGITTAL ONLY (mm)'] = 'Skipped'

    # ── ANS to SPF (SUMMED) ──────────────────────────────────────
    if not ask_skip('ANS to SPF (SUMMED 3D)'):
        if ans_coordinates is not None and spf_inf_coords is not None:
            ans_to_spf_3d = np.linalg.norm(np.array(spf_inf_coords) - np.array(ans_coordinates))
            print(f'  ANS to SPF (3D linear): {ans_to_spf_3d:.2f} mm')
            results['ANS to SPF (SUMMED) (mm)'] = f'{ans_to_spf_3d:.2f}'
        else:
            print('  ANS or SPF Inferior Border not loaded (check Step 3).')
            results['ANS to SPF (SUMMED) (mm)'] = 'Skipped (ANS or SPF not loaded)'
    else:
        results['ANS to SPF (SUMMED) (mm)'] = 'Skipped'

    # ── SPG to CHEEK - AXIAL ─────────────────────────────────────
    if not ask_skip('SPG to CHEEK - AXIAL'):
        node, curve_name = prompt_and_get_node('Cheek curve - Axial plane')
        if node is not None:
            _, _, distance = find_closest_point(SPG, curve_name, axis_xyz=2)
            print(f'  SPG to Cheek (axial): {distance:.2f} mm')
            results['SPG to CHEEK - AXIAL (mm)'] = f'{distance:.2f}'
        else:
            results['SPG to CHEEK - AXIAL (mm)'] = 'Skipped'
    else:
        results['SPG to CHEEK - AXIAL (mm)'] = 'Skipped'

    # ── SPG to CHEEK - CORONAL ───────────────────────────────────
    if not ask_skip('SPG to CHEEK - CORONAL'):
        node, curve_name = prompt_and_get_node('Cheek curve - Coronal plane')
        if node is not None:
            _, _, distance = find_closest_point(SPG, curve_name, axis_xyz=1)
            print(f'  SPG to Cheek (coronal): {distance:.2f} mm')
            results['SPG to CHEEK - CORONAL (mm)'] = f'{distance:.2f}'
        else:
            results['SPG to CHEEK - CORONAL (mm)'] = 'Skipped'
    else:
        results['SPG to CHEEK - CORONAL (mm)'] = 'Skipped'

    # ── SPF to NASAL FLOOR - CORONAL ─────────────────────────────
    if not ask_skip('SPF to NASAL FLOOR - CORONAL'):
        if spf_inf_coords is not None and nasal_floor_coords is not None:
            si_distance = abs(spf_inf_coords[2] - nasal_floor_coords[2])
            print(f'  SPF Inferior Border to Nasal Floor (S/I): {si_distance:.2f} mm')
            results['SPF to NASAL FLOOR - CORONAL (mm)'] = f'{si_distance:.2f}'
        else:
            print('  SPF Inferior Border or Nasal Floor not loaded (check Step 3).')
            results['SPF to NASAL FLOOR - CORONAL (mm)'] = 'Skipped (SPF or Nasal Floor not loaded)'
    else:
        results['SPF to NASAL FLOOR - CORONAL (mm)'] = 'Skipped'

    # ── SPG to NASAL CAVITY - AXIAL ──────────────────────────────
    if not ask_skip('SPG to NASAL CAVITY - AXIAL'):
        node, curve_name = prompt_and_get_node('Nasal Cavity curve - Axial plane (e.g. nasal_cavity_ax)')
        if node is not None:
            _, _, distance = find_closest_point(SPG, curve_name, axis_xyz=2)
            print(f'  SPG to Nasal Cavity (axial): {distance:.2f} mm')
            results['SPG to NASAL CAVITY - AXIAL (mm)'] = f'{distance:.2f}'
        else:
            results['SPG to NASAL CAVITY - AXIAL (mm)'] = 'Skipped'
    else:
        results['SPG to NASAL CAVITY - AXIAL (mm)'] = 'Skipped'

    # ── NASAL SEPTUM to MEDIAL MIDDLE TURBINATE - AXIAL ──────────
    if not ask_skip('NASAL SEPTUM to MEDIAL MIDDLE TURB - AXIAL'):
        _, septum_curve_name = prompt_and_get_node('Nasal Septum curve - Axial plane')
        _, mt_curve_name     = prompt_and_get_node('Middle Turbinate medial curve - Axial plane')
        if septum_curve_name and mt_curve_name:
            pt_septum, pt_mt, min_dist = closest_distance_between_curves(septum_curve_name, mt_curve_name)
            print(f'  Closest point on septum:           {pt_septum}')
            print(f'  Closest point on middle turbinate: {pt_mt}')
            print(f'  Nasal Septum to Middle Turbinate (shortest): {min_dist:.2f} mm')
            results['NASAL SEPTUM to MIDDLE MEDIAL TURB - AXIAL (mm)'] = f'{min_dist:.2f}'
        else:
            results['NASAL SEPTUM to MIDDLE MEDIAL TURB - AXIAL (mm)'] = 'Skipped'
    else:
        results['NASAL SEPTUM to MIDDLE MEDIAL TURB - AXIAL (mm)'] = 'Skipped'

    # ── SPF INF (Lateral Nasal Wall) to MEDIAL MIDDLE TURB - CORONAL ─
    if not ask_skip('SPF INF (Lateral Nasal Wall) to MEDIAL MIDDLE TURB - CORONAL'):
        if spf_inf_coords is not None:
            print(f'  Using SPF Inferior Border: {spf_inf_coords}')
            node, mt_coronal_curve_name = prompt_and_get_node('Middle Turbinate medial surface curve - Coronal plane')
            if node is not None:
                mt_curve_points = get_curve(mt_coronal_curve_name)
                z_diffs = np.abs(mt_curve_points[:, 2] - spf_inf_coords[2])
                matched_point = mt_curve_points[np.argmin(z_diffs)]

                matched_node = slicer.vtkMRMLMarkupsFiducialNode()
                slicer.mrmlScene.AddNode(matched_node)
                matched_node.AddControlPoint(matched_point.tolist())
                matched_node.SetName('MT Coronal point matched to SPF inferior border')

                x_distance = abs(spf_inf_coords[0] - matched_point[0])
                print(f'  Matched MT point (same S/I level): {matched_point}')
                print(f'  SPF Inf to Middle Turbinate (X/lateral): {x_distance:.2f} mm')
                results['SPF INF (Lateral Nasal Wall) to MEDIAL MIDDLE TURB - CORONAL (mm)'] = f'{x_distance:.2f}'
            else:
                results['SPF INF (Lateral Nasal Wall) to MEDIAL MIDDLE TURB - CORONAL (mm)'] = 'Skipped'
        else:
            print('  SPF Inferior Border not loaded (check Step 3).')
            results['SPF INF (Lateral Nasal Wall) to MEDIAL MIDDLE TURB - CORONAL (mm)'] = 'Skipped (SPF not loaded)'
    else:
        results['SPF INF (Lateral Nasal Wall) to MEDIAL MIDDLE TURB - CORONAL (mm)'] = 'Skipped'

    # ── INFERIOR TURB SUPERIOR SURFACE to NASAL FLOOR ────────────
    if not ask_skip('INFERIOR TURB, SUPERIOR to NASAL FLOOR'):
        if nasal_floor_coords is None:
            print('  Nasal Floor not loaded (check Step 3).')
            results['INFERIOR TURB, SUPERIOR to NASAL FLOOR (mm)'] = 'Skipped (nasal floor not loaded)'
        else:
            node, it_sup_curve_name = prompt_and_get_node('Inferior Turbinate superior surface curve - Coronal plane')
            if node is not None:
                it_sup_points       = get_curve(it_sup_curve_name)
                most_superior_point = it_sup_points[np.argmax(it_sup_points[:, 2])]

                ms_node = slicer.vtkMRMLMarkupsFiducialNode()
                slicer.mrmlScene.AddNode(ms_node)
                ms_node.AddControlPoint(most_superior_point.tolist())
                ms_node.SetName('Inferior Turbinate - most superior point')

                si_distance = abs(most_superior_point[2] - nasal_floor_coords[2])
                print(f'  Most superior point on inferior turbinate: {most_superior_point}')
                print(f'  Inferior Turbinate Superior to Nasal Floor (S/I): {si_distance:.2f} mm')
                results['INFERIOR TURB, SUPERIOR to NASAL FLOOR (mm)'] = f'{si_distance:.2f}'
            else:
                results['INFERIOR TURB, SUPERIOR to NASAL FLOOR (mm)'] = 'Skipped'
    else:
        results['INFERIOR TURB, SUPERIOR to NASAL FLOOR (mm)'] = 'Skipped'

    # ── MIDDLE TURB INF SURFACE to NASAL FLOOR - CORONAL ─────────
    if not ask_skip('MIDDLE TURB, INF SURFACE to NASAL FLOOR - CORONAL'):
        if nasal_floor_coords is None:
            print('  Nasal Floor not loaded (check Step 3).')
            results['MIDDLE TURB, INF SURFACE to NASAL FLOOR - CORONAL (mm)'] = 'Skipped (nasal floor not loaded)'
        else:
            node, imt_inf_curve_name = prompt_and_get_node('Inferior Middle Turbinate surface curve - Coronal plane')
            if node is not None:
                imt_inf_points      = get_curve(imt_inf_curve_name)
                most_inferior_point = imt_inf_points[np.argmin(imt_inf_points[:, 2])]

                mi_node = slicer.vtkMRMLMarkupsFiducialNode()
                slicer.mrmlScene.AddNode(mi_node)
                mi_node.AddControlPoint(most_inferior_point.tolist())
                mi_node.SetName('Middle Turbinate - most inferior point')

                si_distance = abs(most_inferior_point[2] - nasal_floor_coords[2])
                print(f'  Most inferior point on middle turbinate: {most_inferior_point}')
                print(f'  Middle Turb Inf Surface to Nasal Floor (S/I): {si_distance:.2f} mm')
                results['MIDDLE TURB, INF SURFACE to NASAL FLOOR - CORONAL (mm)'] = f'{si_distance:.2f}'
            else:
                results['MIDDLE TURB, INF SURFACE to NASAL FLOOR - CORONAL (mm)'] = 'Skipped'
    else:
        results['MIDDLE TURB, INF SURFACE to NASAL FLOOR - CORONAL (mm)'] = 'Skipped'

    # ── INF FORAMEN ROTUNDUM to SPF - CORONAL (Sup/Inf + Summed) ───
    if not ask_skip('INF FORAMEN ROTUNDUM to SPF - CORONAL (both Sup/Inf and Summed)'):
        if spf_inf_coords is not None and fr_coords is not None:
            si_distance     = abs(spf_inf_coords[2] - fr_coords[2])
            linear_distance = np.linalg.norm(np.array(spf_inf_coords) - fr_coords)
            print(f'  Foramen Rotundum: {fr_coords}')
            print(f'  SPF Inferior Border: {spf_inf_coords}')
            print(f'  FR to SPF (S/I only): {si_distance:.2f} mm')
            print(f'  FR to SPF (3D linear): {linear_distance:.2f} mm')
            results['INF FORAMEN ROTUNDUM to SPF - CORONAL, Sup/Inf (mm)'] = f'{si_distance:.2f}'
            results['INF FORAMEN ROTUNDUM to SPF - CORONAL, Summed (mm)']  = f'{linear_distance:.2f}'
        else:
            print('  Foramen Rotundum or SPF not loaded (check Step 3).')
            results['INF FORAMEN ROTUNDUM to SPF - CORONAL, Sup/Inf (mm)'] = 'Skipped (FR or SPF not loaded)'
            results['INF FORAMEN ROTUNDUM to SPF - CORONAL, Summed (mm)']  = 'Skipped (FR or SPF not loaded)'
    else:
        results['INF FORAMEN ROTUNDUM to SPF - CORONAL, Sup/Inf (mm)'] = 'Skipped'
        results['INF FORAMEN ROTUNDUM to SPF - CORONAL, Summed (mm)']  = 'Skipped'

    # ── ORBITAL FLOOR @ PPF to SPF - CORONAL (Sup/Inf + Summed) ───
    if not ask_skip('ORBITAL FLOOR @ PPF to SPF - CORONAL (both Sup/Inf and Summed)'):
        if spf_inf_coords is not None:
            node, orbital_floor_curve_name = prompt_and_get_node('Orbital Floor curve - Coronal plane')
            if node is not None:
                orbital_floor_points = get_curve(orbital_floor_curve_name)
                most_inferior_point  = orbital_floor_points[np.argmin(orbital_floor_points[:, 2])]

                of_node = slicer.vtkMRMLMarkupsFiducialNode()
                slicer.mrmlScene.AddNode(of_node)
                of_node.AddControlPoint(most_inferior_point.tolist())
                of_node.SetName('Orbital Floor - most inferior point')

                si_distance     = abs(most_inferior_point[2] - spf_inf_coords[2])
                linear_distance = np.linalg.norm(most_inferior_point - np.array(spf_inf_coords))
                print(f'  Most inferior point on orbital floor: {most_inferior_point}')
                print(f'  Orbital Floor to SPF (S/I only): {si_distance:.2f} mm')
                print(f'  Orbital Floor to SPF (3D linear): {linear_distance:.2f} mm')
                results['ORBITAL FLOOR @ PPF to SPF - CORONAL, Sup/Inf (mm)'] = f'{si_distance:.2f}'
                results['ORBITAL FLOOR @ PPF to SPF - CORONAL, Summed (mm)']  = f'{linear_distance:.2f}'
            else:
                results['ORBITAL FLOOR @ PPF to SPF - CORONAL, Sup/Inf (mm)'] = 'Skipped'
                results['ORBITAL FLOOR @ PPF to SPF - CORONAL, Summed (mm)']  = 'Skipped'
        else:
            print('  SPF not loaded (check Step 3).')
            results['ORBITAL FLOOR @ PPF to SPF - CORONAL, Sup/Inf (mm)'] = 'Skipped (SPF not loaded)'
            results['ORBITAL FLOOR @ PPF to SPF - CORONAL, Summed (mm)']  = 'Skipped (SPF not loaded)'
    else:
        results['ORBITAL FLOOR @ PPF to SPF - CORONAL, Sup/Inf (mm)'] = 'Skipped'
        results['ORBITAL FLOOR @ PPF to SPF - CORONAL, Summed (mm)']  = 'Skipped'

    # ── IOF MEDIAL BORDER to SPF - CORONAL (Sup/Inf + Summed) ──────
    if not ask_skip('IOF MEDIAL BORDER to SPF - CORONAL (both Sup/Inf and Summed)'):
        if spf_inf_coords is not None and iof_coords is not None:
            si_distance     = abs(iof_coords[2] - spf_inf_coords[2])
            linear_distance = np.linalg.norm(iof_coords - np.array(spf_inf_coords))
            print(f'  IOF medial border: {iof_coords}')
            print(f'  SPF Inferior Border: {spf_inf_coords}')
            print(f'  IOF Medial to SPF (S/I only): {si_distance:.2f} mm')
            print(f'  IOF Medial to SPF (3D linear): {linear_distance:.2f} mm')
            results['IOF MEDIAL to SPF - CORONAL, Sup/Inf (mm)']  = f'{si_distance:.2f}'
            results['IOF MEDIAL to SPF - CORONAL, Summed (mm)']   = f'{linear_distance:.2f}'
        else:
            print('  IOF medial border or SPF not loaded (check Step 3).')
            results['IOF MEDIAL to SPF - CORONAL, Sup/Inf (mm)']  = 'Skipped (IOF or SPF not loaded)'
            results['IOF MEDIAL to SPF - CORONAL, Summed (mm)']   = 'Skipped (IOF or SPF not loaded)'
    else:
        results['IOF MEDIAL to SPF - CORONAL, Sup/Inf (mm)']  = 'Skipped'
        results['IOF MEDIAL to SPF - CORONAL, Summed (mm)']   = 'Skipped'

    # ── LATERAL NASAL WALL THICKNESS @ SPF - CORONAL ─────────────
    if not ask_skip('LATERAL NASAL WALL THICKNESS @ SPF - CORONAL'):
        node, lat_wall_line = prompt_and_get_node('Lateral nasal wall thickness line @ SPF - Coronal plane (e.g. lat_nasal_wall_cor)')
        if node is not None:
            p0, p1 = get_line_points(node)
            lat_wall_thickness = np.linalg.norm(p1 - p0)
            print(f'  Lateral Nasal Wall Thickness @ SPF: {lat_wall_thickness:.2f} mm')
            results['LATERAL NASAL WALL THICKNESS @ SPF - CORONAL (mm)'] = f'{lat_wall_thickness:.2f}'
        else:
            results['LATERAL NASAL WALL THICKNESS @ SPF - CORONAL (mm)'] = 'Skipped'
    else:
        results['LATERAL NASAL WALL THICKNESS @ SPF - CORONAL (mm)'] = 'Skipped'

    # ── MIDLINE HARD PALATE LENGTH - SAGITTAL ────────────────────
    if not ask_skip('MIDLINE HARD PALATE LENGTH - SAGITTAL'):
        node, hard_palate_line = prompt_and_get_node('Midline hard palate line - Sagittal plane (e.g. hard_palate_sag)')
        if node is not None:
            p0, p1 = get_line_points(node)
            hard_palate_length = np.linalg.norm(p1 - p0)
            print(f'  Midline Hard Palate Length: {hard_palate_length:.2f} mm')
            results['MIDLINE HARD PALATE LENGTH - SAGITTAL (mm)'] = f'{hard_palate_length:.2f}'
        else:
            results['MIDLINE HARD PALATE LENGTH - SAGITTAL (mm)'] = 'Skipped'
    else:
        results['MIDLINE HARD PALATE LENGTH - SAGITTAL (mm)'] = 'Skipped'


# ── STEP 5: Summary of all results ──────────────────────────────
print('\n' + '=' * 60)
print(' RESULTS SUMMARY')
print('=' * 60)
print(f'  SPG Centroid: {SPG}')
print()
for label, value in results.items():
    print(f'  {label}: {value}')
print('=' * 60)
print(' Done!')
print('=' * 60)
