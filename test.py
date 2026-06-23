from gnr import TestingApplication

if __name__ == "__main__":
    app = TestingApplication()

    print("Setup testing app")
    app.test_all()

