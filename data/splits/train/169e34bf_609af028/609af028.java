import java.io.*;
import java.util.*;

public class new1{
	
	public static void main(String[] args)  throws IOException{
    	FastReader s = new FastReader();
    	//long l1 = System.currentTimeMillis();
    	BufferedWriter output = new BufferedWriter(new OutputStreamWriter(System.out));
    	int t =  s.nextInt();
    	for(int z = 0; z < t; z++) {
    		int n = s.nextInt();
    		int[] arr = new int[n];
    		ArrayList<Integer> blue = new ArrayList<Integer>();
    		ArrayList<Integer> red = new ArrayList<Integer>();
    		for(int i = 0; i < n; i++) {
    			arr[i] = s.nextInt();
    		}
    		String str = s.next();
    		for(int i = 0; i < n; i++) {
    			if(str.charAt(i) == 'B') blue.add(arr[i]);
    			else red.add(arr[i]);
    		}
    		red.sort(null); blue.sort(null);
    		int j = red.size() - 1; int k = blue.size() - 1;
    		while(j >= 0 && red.get(j) > n) j--;
    		boolean ans = true;
//    		System.out.println(blue.toString());
//    		System.out.println(red.toString());
    		for(int m = n; m >= 1; m--) {
    			if(j >= 0) j--;
    			else if(k >= 0 && blue.get(k) >= m) k--;
    			else ans = false;
    			//System.out.println(m + "  " + j + " " + k);
    			while(j >= 0 && red.get(j) > m - 1) j--;
    		}
    		if(ans) System.out.println("YES");
    		else System.out.println("NO");
    		
    		
    	}
    	//output.write((System.currentTimeMillis() - l1) + "\n");
    	//output.flush();	
    }
}



