import numpy as np
import lle_calcs
from get_unifaclle_parameters import get_unifac_parameters

# Dummy placeholder — replace with your real LLE function
def run_lle_calculation(comp, z, T):
    try:
        params = get_unifac_parameters(comp)
        N_I, N_II, xI, xII, vv = lle_calcs.lle_calc(comp, z, T, 'unifac-lle', params)
        return {'x_I': xI, 'x_II': xII}
    except:
        return None  # If it fails, skip


benchmark_cases = [
    {
        'name': 'System 1 - T = 293.2 K',
        'T': 293.2,
        'components': ['WATER', 'ACETIC_ACID', 'ETHYL_ACETATE'],
        'tolerance': 1e-3,
        'feed': [
            [0.7922, 0.0212, 0.1866],
            [0.7909, 0.0352, 0.1739],
            [0.7895, 0.0523, 0.1582]
        ],
        'expected_x_I': [
            [0.1673, 0.052795, 0.77991],
            [0.20392, 0.087439, 0.70864],
            [0.25078, 0.12762, 0.6216]
        ],
        'expected_x_II': [
            [0.98252, 0.011577, 0.0059054],
            [0.97454, 0.018857, 0.0066064],
            [0.96466, 0.027811, 0.0075282]
        ]
    },
    {
        'name': 'System 2 - T = 293.2 K',
        'T': 293.2,
        'components': ['WATER', 'ACETIC_ACID', 'BUTYL_ACETATE'],
        'tolerance': 1e-3,
        'feed': [
            [0.7956, 0.0368, 0.1676],
            [0.7808, 0.0705, 0.1487],
            [0.7679, 0.1067, 0.1254]
        ],
        'expected_x_I': [
            [0.11467, 0.09397, 0.79136],
            [0.15241, 0.17052, 0.67707],
            [0.19888, 0.24304, 0.55808]
        ],
        'expected_x_II': [
            [0.97802, 0.021484,	0.00049768],
            [0.95678, 0.042489,	0.00073013],
            [0.93133, 0.067542,	0.0011252]
        ]
    },
    {
        'name': 'System 3 - T = 293.2 K',
        'T': 293.2,
        'components': ['WATER', 'ACETIC_ACID', 'ISOBUTYL_ACETATE'],
        'tolerance': 1e-3,
        'feed': [
            [0.7886, 0.0707, 0.1407],
            [0.7795, 0.0997, 0.1208],
            [0.7638, 0.1318, 0.1044]
        ],
        'expected_x_I': [
            [0.15411, 0.17386, 0.67203],
            [0.19231, 0.23423, 0.57346],
            [0.23625, 0.28905, 0.4747]
        ],
        'expected_x_II': [
            [0.95573, 0.043526, 0.00074335],
            [0.93482, 0.064115, 0.0010618],
            [0.91029, 0.088136, 0.0015768]
        ]
    },
    # Add more systems here if needed...
]


def run_tests():
    for case in benchmark_cases:
        print(f"\n🔍 Testing: {case['name']}")
        T = case['T']
        components = case['components']
        tol = case['tolerance']
        feeds = case['feed']
        x_I_refs = case['expected_x_I']
        x_II_refs = case['expected_x_II']

        for i in range(len(feeds)):
            feed = np.array(feeds[i])
            x_I_ref = np.array(x_I_refs[i])
            x_II_ref = np.array(x_II_refs[i])

            try:
                result = run_lle_calculation(components, feed, T)
                x_I_calc = np.array(result['x_I'])
                x_II_calc = np.array(result['x_II'])

                # Ensure phase with higher water content is x_I
                if x_I_calc[0] > x_II_calc[0]:
                    x_I_calc, x_II_calc = x_II_calc, x_I_calc

                pass_I = np.allclose(x_I_calc, x_I_ref, atol=tol)
                pass_II = np.allclose(x_II_calc, x_II_ref, atol=tol)

                if pass_I and pass_II:
                    print(f"✅ Tie line {i+1}: PASS")
                else:
                    print(f"❌ Tie line {i+1}: FAIL")
                    print(f"  Expected x_I:  {x_I_ref}, Got: {x_I_calc}")
                    print(f"  Expected x_II: {x_II_ref}, Got: {x_II_calc}")

            except Exception as e:
                print(f"⚠️  Tie line {i+1}: ERROR - {e}")


if __name__ == "__main__":
    run_tests()
