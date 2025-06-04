import java.io.*;
import java.util.*;

public class GFG {
    
    
    
	public static void main (String[] args) {
	    Scanner sc = new Scanner(System.in);
	    int t = sc.nextInt();
	     while(t-- > 0){
	         int n = sc.nextInt();
	         long pre = 0,ans = Long.MAX_VALUE;
	         long[] arr = new long[n];
	         PriorityQueue<Long> epq = new PriorityQueue<>();
	         PriorityQueue<Long> opq = new PriorityQueue<>();
	         for(int i=0;i<n;i++)
	         {
	             arr[i] = sc.nextLong();
	             if(i%2==0)
	             epq.add(arr[i]);
	             else
	             opq.add(arr[i]);
	             pre+=arr[i];
	             if(i>0)
	             ans = Math.min(ans,pre+(n-epq.size())*epq.peek()+(n-opq.size())*opq.peek());
	         }
	         System.out.println(ans);
	     }    
	}
}
