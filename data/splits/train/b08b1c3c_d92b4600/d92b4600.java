import java.util.Scanner;

public class D_724 {

	@SuppressWarnings("resource")
	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
		int t = input.nextInt();
		for(int test = 0; test < t; test++){
			int n = input.nextInt();
			ListNode on = new ListNode(input.nextInt(), null, null);
			boolean good = true;
			for(int i = 1; i < n; i++){
				int num = input.nextInt();
				if(good){
					int at = on.data;
					if(num > at){
						if(on.next == null || num < on.next.data){
							on.next = new ListNode(num, on, on.next);
							on = on.next;
							if(on.next != null){
								on.next.prev = on;
							}
						}else if(num == on.next.data){
							on = on.next;
						}else{
							good = false;
						}
					}else if(num < at){
						if(on.prev == null || num > on.prev.data){
							on.prev = new ListNode(num, on.prev, on);
							on = on.prev;
							if(on.prev != null){
								on.prev.next = on;
							}
						}else if(num == on.prev.data){
							on = on.prev;
						}else{
							good = false;
						}
					}
				}
			}
			if(good){
				System.out.println("YES");
			}else{
				System.out.println("NO");
			}
		}
	}

	static class ListNode{

		int data;
		ListNode prev;
		ListNode next;

		ListNode(int data, ListNode prev, ListNode next){
			this.data = data;
			this.prev = prev;
			this.next = next;
		}

	}

}
