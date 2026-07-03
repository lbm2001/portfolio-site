import SiteHeader from "@/components/SiteHeader";
import About from "@/components/About";
import Work from "@/components/Work";
import Blog from "@/components/Blog";
import CvStrip from "@/components/CvStrip";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <main>
      <SiteHeader />
      <About />
      <Work />
      <Blog />
      <CvStrip />
      <Footer />
    </main>
  );
}
