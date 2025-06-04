import java.io.PrintWriter;
import java.util.*;
public class codeforces {
	public static void main(String[] args) {
		PrintWriter out=new PrintWriter(System.out);
    	Scanner s=new Scanner(System.in);  
    	int t=s.nextInt();
    	for(int tt=0;tt<t;tt++) {
    		int n=s.nextInt(),m=s.nextInt(),x=s.nextInt();
    		HashMap<Integer,ArrayList<Integer>> map=new HashMap<>();
    		int a[]=new int[n];
    		for(int i=0;i<n;i++) {
    			a[i]=s.nextInt();
    			if(map.containsKey(a[i])) {
    				ArrayList<Integer> c=map.get(a[i]);
    				c.add(i+1);
    				map.put(a[i], c);
    			}else {
    				ArrayList<Integer> c=new ArrayList<>();
    				c.add(i+1);
    				map.put(a[i], c);
    			}
    		}
    		sort(a);
    		long ans[]=new long[m];
    		int c=0;
    		boolean l=true;
    		for(int i=n-1;i>=0;i--) {
    			if(c==m && l) {
    				c=m-1;
    				l=false;
    			}else if(!l && c==-1){
    				l=true;
    				c=0;
    			}
    			ans[c]+=a[i];
    			if(l) {
    				c++;
    			}else {
    				c--;
    			}
    		}
    		long min=Integer.MAX_VALUE,max=Integer.MIN_VALUE;
    		for(int i=0;i<m;i++) {
    			min=Math.min(ans[i], min);
    			max=Math.max(ans[i], max);
    		}
    		if(max-min>x) {
    			System.out.println("NO");
    		}else {
    			System.out.println("YES");
    			int a1[]=new int[n];
    			c=0;
    			l=true;
    			for(int i=n-1;i>=0;i--) {
    				if(c==m && l) {
        				c=m-1;
        				l=false;
        			}else if(!l && c==-1){
        				l=true;
        				c=0;
        			}
    				a1[map.get(a[i]).get(0)-1]=c+1;
    				
    				map.get(a[i]).remove(0);
    				if(l) {
        				c++;
        			}else {
        				c--;
        			}
    				
    			}
    			for(int i=0;i<n;i++) {
    				System.out.print(a1[i]+" ");
    			}
    			System.out.println();
    		}
    	}
    	out.close();
	    s.close();
	}
	
	static void sort(int[] a) {
		ArrayList<Integer> l=new ArrayList<>();
		for (int i:a) l.add(i);
		Collections.sort(l);
		for (int i=0; i<a.length; i++) a[i]=l.get(i);
	}
	
}
