import Nav from "@/components/Nav";
import Hero from "@/components/Hero";
import About from "@/components/About";
import Work from "@/components/Work";
import Blog from "@/components/Blog";
import CvStrip from "@/components/CvStrip";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <main>
      <Nav />
      <Hero />
      <About />
      <CvStrip />
      <Work />
      <Blog />
      <Footer />
    </main>
  );
}
