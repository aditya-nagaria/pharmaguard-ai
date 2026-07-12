from pharma_guard import app
import os

# Test scenarios - task description + expected minimum frameworks
test_cases = [
    {
        "name": "LIMS Release with E-signatures",
        "task": "Releasing a new LIMS module that handles electronic batch records with electronic signatures and audit trail capabilities.",
        "expected_frameworks": ["21 CFR Part 11", "GxP"],
        "expected_risk_min": "medium"
    },
    {
        "name": "BOM Revision Before Release",
        "task": "BOM is pending revision, product scheduled to release in 2 days.",
        "expected_frameworks": ["Change Control"],
        "expected_risk_min": "medium"
    },
    {
        "name": "Custom Software Validation",
        "task": "We built a custom application to control lab equipment and need to validate it before go_live.",
        "expected_frameworks": ["GAMP 5"],
        "expected_risk_min": "medium"
    },
    {
        "name": "Medical Device Software Development",
        "task": "Developing embedded software for a Class II medical device that monitors patient vitals.",
        "expected_frameworks": ["IEC 62304"],
        "expected_risk_min": "medium"
    },
    {
        "name": "FDA AUdit Preparation",
        "task": "We have an FDA audit next week and need to confirm our documentation is ready.",
        "expected_frameworks": ["Audit Readiness"],
        "expected_risk_min": "low"
    }
]

risk_order = {"low": 0, "medium":1, "high": 2}

def run_tests():
    print("=" * 60)
    print("PHARMAGUARD AI -TEST SUITE")
    print("=" *60)

    passed = 0
    failed = 0
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"Task: {test['task'][:70]}...")

        config = {"configurable": {"thread_id": f"test_{i}"}}

        result = app.invoke({
            "task_description": test["task"],
            "task_type": "",
            "task_scope": "",
            "applicable_frameworks": [],
            "retrieved_requirements": [],
            "deviations": [],
            "risk_level": "",
            "compliance_brief": {},
            "messages": []
        }, config=config)

        actual_frameworks = result["applicable_frameworks"]
        actual_risk = result["risk_level"].lower()

        framework_match = any(
            expected in actual_frameworks for expected in test["expected_frameworks"]
        )

        risk_match = risk_order.get(actual_risk, 0) >= risk_order.get(test["expected_risk_min"],0)

        if framework_match and risk_match:
            print(f" PASS - Frameworks: {actual_frameworks} | Risk: {actual_risk}")
            passed +=1
        else:
            print(f" FAIL - Frameworks: {actual_frameworks} | Risk: {actual_risk}")
            print(f" Expected at least one of: {test['expected_frameworks']} | Min risk: {test['expected_risk_min']}")
            failed +=1

    print()
    print("=" * 60)
    print(f"RESULT: {passed} passed, {failed} failed out of {len(test_cases)}")
    print("=" *60)
    
if __name__ == "__main__":
     run_tests()