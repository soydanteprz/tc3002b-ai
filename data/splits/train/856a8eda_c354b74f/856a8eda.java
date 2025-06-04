import java.io.*;
import java.util.*;

public class three{
	public static class Pair implements Comparable<Pair>{
		int min;
		int idx;
		
	}

public static void main(String[] args) throws Exception {
        MyScanner scn = new MyScanner();
        out = new PrintWriter(new BufferedOutputStream(System.out));
         /*
      int n      = sc.nextInt();        // read input as integer
      long k     = sc.nextLong();       // read input as long
      double d   = sc.nextDouble();     // read input as double
      String str = sc.next();           // read input as String
      String s   = sc.nextLine();       // read whole line as String
 
      int result = 3*n;
      out.println(result);                    // print via PrintWriter
      */

      //The Code Starts here
      int t = scn.nextInt();
      while(t-- > 0){
      	int n = scn.nextInt();
      	int m = scn.nextInt();

      	int x = scn.nextInt();
      	int arr[] = scn.nextIntArray(n);
      	PriorityQueue<Pair> pq = new PriorityQueue<>();
      	      	System.out.println("YES");

      	for(int i=0;i<m;i++){
      		Pair p = new Pair();
      		p.min = arr[i];
      		p.idx = i+1;
      		pq.add(p);
      		System.out.print(p.idx + " ");
      	}

      	for(int i=m;i<n;i++){
      		Pair p = pq.peek();
      		int mini = p.min;
      		int index = p.idx;
      		System.out.print(index + " ");
      		pq.remove();
      		Pair np = new Pair();
      		np.min = arr[i] + mini;
      		np.idx = index;
      		pq.add(np);
      	}
      	System.out.println();




      	
      }





      //The Code Ends here
        out.close();

    

    
 }





   //-----------PrintWriter for faster output---------------------------------
    public static PrintWriter out;
 
    //-----------MyScanner class for faster input----------
    
 
    
 
    // public static class Pair implements Comparable<Pair> {
    //     long u;
    //     long v;
 
    //     public Pair(long u, long v) {
    //         this.u = u;
    //         this.v = v;
    //     }
 
 
    //     public int hashCode() {
    //         int hu = (int) (u ^ (u >>> 32));
    //         int hv = (int) (v ^ (v >>> 32));
    //         return 31 * hu + hv;
    //     }
 
    //     public boolean equals(Object o) {
    //         Pair other = (Pair) o;
    //         return u == other.u && v == other.v;
    //     }
 
    //     public int compareTo(Pair other) {
    //         return Long.compare(u, other.u) != 0 ? Long.compare(u, other.u) : Long.compare(v, other.v);
    //     }
 
    //     public String toString() {
    //         return "[u=" + u + ", v=" + v + "]";
    //     }
    // }
    //--------------------------------------------------------
}
