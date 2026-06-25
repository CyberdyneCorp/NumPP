from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout


class NumPPConan(ConanFile):
    name = "numpp"
    version = "1.0.0"
    license = "MIT"
    author = "Leonardo Araujo dos Santos"
    url = "https://github.com/leonardoaraujosantos/NumPP"
    description = (
        "Modern C++20 port of NumPy with tiered BLAS/LAPACK/GPU acceleration "
        "and a portable CPU fallback that compiles on iOS/Android."
    )
    topics = ("numpy", "ndarray", "linear-algebra", "cpp20", "blas", "gpu")

    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_blas": [True, False],
        "with_lapack": [True, False],
    }
    # Core has zero required dependencies; all acceleration is opt-in and OFF by default.
    default_options = {"shared": True, "fPIC": True, "with_blas": False, "with_lapack": False}

    exports_sources = "CMakeLists.txt", "cmake/*", "include/*", "src/*", "LICENSE"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["NUMPP_BUILD_TESTS"] = "OFF"
        tc.variables["NUMPP_WITH_BLAS"] = self.options.with_blas
        tc.variables["NUMPP_WITH_LAPACK"] = self.options.with_lapack
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["numpp"]
