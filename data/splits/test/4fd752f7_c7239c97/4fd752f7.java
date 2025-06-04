import java.io.*;
import java.util.*;

public class C {
	
	public static void main(String[] args) {
        FastScanner in = new FastScanner();
        PrintWriter out = new PrintWriter(System.out);
        int t = in.nextInt(), tt = 0;
        while(t-->0) { 
        	int n = in.nextInt(), m = in.nextInt(), x = in.nextInt();
        	int aa[] = in.readArray(n);
        	Pair a[] = new Pair[n];
        	for(int i=0;i<n;i++) a[i] = new Pair(i,aa[i]);
        	PriorityQueue<Pair> pq = new PriorityQueue<>();
        	Arrays.sort(a);
        	int goneTo[] = new int[n];
        	for(int i=0;i<m;i++){
        		pq.add(new Pair(i+1,a[n-1-i].y));
        		goneTo[a[n-1-i].x] = i+1;
        	}
        	for(int i=n-m-1;i>=0;i--){
        		Pair xx = pq.poll();
        		xx.y += a[i].y;
        		goneTo[a[i].x] = xx.x;
        		pq.add(new Pair(xx.x,xx.y));
        	}
        	Iterator<Pair> it = pq.iterator();
        	ArrayList<Pair> arr = new ArrayList<>();
        	while(it.hasNext()) arr.add(it.next());
        	Collections.sort(arr);
        	//for(Pair i : arr) out.print(i.y+" ");
        	//out.println();
        	if(arr.get(m-1).y-arr.get(0).y>x) out.println("NO");
        	else{
        		out.println("YES");
        		int pos[] = new int[n];
        		for(Pair i : a) pos[i.x] = goneTo[i.x];
        		for(int i : pos) out.print(i+" ");
        		out.println();
        	}
    		//tt++; out.println("Case #"+tt+": "+ans);
        }
        out.flush();
}

static class Pair implements Comparable<Pair> {
		int x, y;
		Pair(int a, int b){ x = a; y = b; }		
		
		}	

	
	

	static final Random random = new Random();

	
}
