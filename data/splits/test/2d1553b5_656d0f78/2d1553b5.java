import java.util.*;
import java.io.*; 
public class Interactive{     
    static long MOD=(long)1e9+7;
    //debug
    
    static Scanner sc = new Scanner(System.in);
    //Global variables and functions
    static long query(int l, int r) {
    	System.out.println("? " + l + " " + r);
    	return sc.nextLong();
    }
    static int query(int i) {
    	System.out.println("? " + i);
    	return sc.nextInt();
    }
    //Main function(The main code starts from here)
    public static void main (String[] args) throws java.lang.Exception {
    	int test=1;
    	test=sc.nextInt();
    	while(test-->0) {
    		int n = sc.nextInt();
    		int ans[] = new int[n];
    		for(int i = 1; i <= n; i++) {
    			if(ans[i - 1] != 0) continue;
    			HashSet<Integer> set = new HashSet<>();
    			ArrayList<Integer> arr = new ArrayList<>();
    			while(true) {
    				int x = query(i);
    				if(set.contains(x)) break;
    				set.add(x);
    				arr.add(x - 1);
    			}
    			for(int j = 0; j < arr.size(); j++) {
    				ans[arr.get(j)] = arr.get((j + 1) % arr.size()) + 1;
    			}
    		}
    		System.out.print("! ");
    		for(int x: ans) System.out.print(x+" ");
    		System.out.println();
    	}
    }
}
