Prerequisites:
Visual Studio 2022
MSMPI
Microsoft C++

Steps:
Select the Debug and x64 processor in the Visual studio solution configuration dropdowns.
Build the solution from solution explorer by right clicking on the solution and click build.
The executable file is generated in the folder "\source\repos\MPI\X64\Debug"

Run the executable file with the command line arguments
	- ./MPI.exe --oversubscribe -np {number of processors} knn
	- ./MPI.exe --oversubscribe -np 8 knn
