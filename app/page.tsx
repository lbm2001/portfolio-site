import Nav from "@/components/Nav";
import Hero from "@/components/Hero";
import About from "@/components/About";
import Work from "@/components/Work";
import Blog from "@/components/Blog";
import ResumeStrip from "@/components/ResumeStrip";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <main>
      <Nav />
      <Hero />
      <About />
      <ResumeStrip />
      <Work />
      <Blog />
      <Footer />
    </main>
  );
}
