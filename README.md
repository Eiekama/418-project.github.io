# Particle-Based Fluid Simulation
Contributors: Jingxuan Chen, Hank Xu

## Problem Definition
We are going to implement a particle based 2D fluid solver based on smoothed particle hydrodynamics  (SPH).

### Background
At a very high level, in particle based fluid simulation the evolution of a particle’s movement depends on contributions from its surrounding particles (particles that are too far away are ignored). More specifically, at each timestep each particle would compute its local density and pressure based on the position of surrounding particles, and using those two values compute the force acting on the particle, which can be used to derive acceleration and update the particle position. Since each particle updates its own position independently from the rest within a timestep, it would make sense to parallelize across particles to speed up the runtime. 

### The Challenge
Similar to the galaxy evolution problem discussed in class, in particle based fluid simulation the amount of work per particle and communication pattern is non-uniform since it depends on the local density of particles, the particles move so costs and communication patterns change over time, and there is also a lot of locality to exploit since particles close to each other in space require similar data to calculate forces. Thus the goal would be the same, we want equal work per processor and assignment should preserve locality.

However, a key difference is that particle positions in fluid simulation change much faster than those in galaxy evolution, so a similar semi-static assignment approach would be unlikely to yield the same benefits for fluid simulation as more frequent recomputation is needed to maintain the same level of locality. Though, one benefit that we have is that dependencies in fluid simulation are much simpler than dependencies in the galaxy evolution problem (In fluid simulation, particles outside a certain radius are ignored, whereas in galaxy evolution no particles can be ignored and faraway particles have to be averaged instead) so perhaps the reduced compute balances out the previous downside.

### Resources
We will heavily refer to [this](https://lucasschuermann.com/writing/implementing-sph-in-2d) sequential implementation of a SPH solver provided by Lucas V. Schuermann written in C++.

### Platform Choice
We will be using **glfw** for windowing, **OpenGL** for rendering, and using **OpenMP** and **CUDA** to explore GPU and CPU parallelization.

Since high resolution fluid simulation requires a whole lot of particles, it is tempting to have each particle be its own thread and use CUDA to dispatch a large number of threads to run in parallel. However, this approach might have less locality since we cannot control which particle gets assigned to which processor. With OpenMP on the other hand, we have more control over how to distribute data to processors improving workload balance and locality, but we would be unable to leverage the power of a GPU. Since both have its pros and cons we would be interested in trying both approaches.

## Progress
We have set up a basic renderer for our fluid simulation, imported the sequential implementation, and made lots of progress on the CPU parallelization side.

We started with a naive parallelization by simply changing the for loop over particles to a parallel for. Building upon that, we partitioned the simulation space into cells so that each particle only checks particles in its adjacent cells rather than all particles. This reduces computation costs from O(N2) to O(NM). To improve spatial locality, the cells were hashed such that particles in the same cell can be sorted next to each other in the particle list.

GPU implementation wise we have fallen a bit behind due to running into some issues figuring out how to compile CUDA code locally. The naive GPU implementation has been written, but we haven’t gotten a chance to test it just yet. However, we should have recently figured out the compile issues so progress shouldn’t be blocked.

We believe the core deliverables stated in the proposal are all reachable, though we will probably not extend our implementation to 3D. The other stretch goal that we have (the one about exploring optimizations unique to the fluid simulation problem) is kind of already partially fulfilled since partitioning the simulation space into a grid was one of the optimizations.

### Planned Presentation
We will have a demo showcasing an interactive application where one can test out different kinds of parallelism and see the fluid simulation happen in real time. There will be real time graphs showing profiled frame times and static graph showing cache locality or profiled statistics.

### Preliminary Results
**CPU Results** <br>
Particle Count: 40K <br>
Acceleration Method: spatial compact hashing <br>
Video Speedup: 4x <br>
[![](https://markdown-videos-api.jorgenkh.no/youtube/5p2VmZYt70g)](https://youtu.be/5p2VmZYt70g)

You can see the renderer starts off fast, but when particles clump together, it gets really slow (somewhere in the middle of the video). This is most likely due to load imbalance. We will tackle this using cost zones and fine grained parallelism.

## Goals and Deliverables

### Planned Goals

#### Minimum Goals
- Sequential implementation of simulation using C++ for baseline comparisons
- Per-particle parallelization of simulation using CUDA
- Naive implementation (random static assignment) using OpenMP

#### Remaining Goals
- Barnes-Hut like implementation using OpenMP
- Analytic graphs such as speedup and cache misses for each approach
- Basic renderer to showcase simulation

### Stretch Goals
- Explore commonly used optimization techniques for SPH (e.g. grid system, solver term precomputation) and integrate at least one that’s non trivial to implement into our approaches
- ~~Make simulation 3D~~

## Schedule
| Timeline | Todo |
| -------- | ---- |
| Mar 26 - Mar 30 (Week 0) | Setup basic application framework and renderer.<br>Port sequential code.<br>Setup timing functions. |
| Mar 31 - Apr 6 (Week 1) | Implement naive CPU solver.<br>~~Implement GPU solver.~~ |
| Apr 7 - April 13 (Week 2) | Spatial binning. |
| Apr 14 - Apr 15 (Week 3) (Milestone Report due) | Gather analytic data.<br>Write report. |
| Apr 16 - Apr 20 (Week 3) | Week 3.1: Explore fine grained parallelism with OpenMP tasks and cost zones. (Hank)<br>Week 3.1: GPU solver with naive per particle parallelism (Jingxuan)<br>Week 3.2: Exploit more locality by assigning threads to cells, not particles (Hank)<br>Week 3.2: Integrate spatial binning into GPU solver (Jingxuan) |
| Apr 21 - Apr 27 (Week 4) | Week 4.1: Build UI (ImGui) for testing out different methods of parallelism for demo purposes (Hank)<br>Week 4.1: Explore if there’s ways that give better locality than what’s currently implemented to communicate data between CPU and GPU (Jingxuan)<br>Week 4.2: Project writeup, analytics data, make poster, write report. (Hank + Jingxuan) |
| Apr 28 (Final Report due) | Update analytic data if necessary.<br>Write report.<br>Make poster. |