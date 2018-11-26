---
classes: wide
---

A number of years ago, I created some software for expediting the processing of halo-finding simulations. It formed part of my M.S thesis at the time, so I thought I'd give it a quick write-up for anyone with a passing interest. In addition, it should be in a state that would allow someone to tailor it to their own requirements with minimal effort.

### What is halo-finding?

Cosmological simulations are used by astronomers to investigate large scale structure formation and galaxy evolution. Halo-finding is the discovery of gravitationally-bound objects such as dark matter halos, and it is a crucial step in this process. Okay, but what is a [dark matter halo](https://en.wikipedia.org/wiki/Dark_matter_halo)? Well firstly, dark matter refers to transparent material that is postulated to exist in space, and accounts for the majority of the total mass in the universe. It cannot be observed directly, via telescope or other means, nor does it emit or absorb light or any other electro-magnetic radiation at any significant level. Instead, it's existence is inferred from its gravitational effects on visible matter.

![dark-matter-halo](../assets/blog/dark-matter-halo.png){:height="350px" width="350px" .align-center}
<figcaption style="width:70%;margin:auto;padding-bottom:20px;text-align:center;">A simplistic impression of a dark matter halo encompassing a galaxy.</figcaption>

A dark matter halo, then, is an almost spherical component of a galaxy, and is responsible for holding the galaxy together via gravity. Astronomers believe that this structure is closely related to galaxy evolution, and so it is no surprise that many means for its detection have been developed.

### Halo-finding methods

Perhaps the simplest and easiest to implement halo-finding method - and, totally incidentally, the one we use - is known as the friends-of-friends method. Essentially, the method aims to identify overly-dense objects in observed - or simulated - galaxy distributions. This is done by selecting members, or particles, that lie roughly within a local iso-density contour, which is determined by a free parameter called the linking length. Essentially, the method links together particles that are within a certain distance of one another, according to a threshold called the linking length.

![dark-matter-halo](../assets/blog/friends-of-friends-method.png){: .align-center}
<figcaption style="width:70%;margin:auto;padding-bottom:20px;text-align:center;">Two distinct halos with particles linked by the friends-of-friends method. Particles are only linked if the distance between them is less than a predefined linking length.</figcaption>
