/* 쇼핑몰 상품 목록 페이지 */
page ProductPage {

  /* 자연어와 함수형 스타일이 섞인 고수준 정의 */
  component ProductGrid {
    
    container: flexbox(wrap: true, gap: "1rem")

    /* 반복문 정의 */
    item: ProductCard foreach $products {
      image: $product.hero_image
      title: $product.name
      price: $product.current_price
      rating: stars($product.avg_rating)
      
      /* AI가 해석할 자연어 스타일링 지침 */
      styling: "Large card with shadow, rounded corners. Clean white background."
      
      hover_effects: {
        transform: "scale(1.05)"
        transition: "Smooth generic transition"
      }
      
      button: WishlistButton {
        position: "top-right absolute"
        style: "Small heart icon, red when active, otherwise gray outline"
      }
    }
  }

  component SidebarFilters {
    layout: "Vertical sidebar on left, hidden on mobile"
    items: [ "Category", "Price Range", "Colors" ]
  }
}