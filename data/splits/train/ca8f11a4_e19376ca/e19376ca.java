import java.io.*;
import java.util.*;

public class new1{

    
    public static int count(int[] arr, int[] dest) {
    	int n = arr.length;
    	int count = 0;
    	for(int i = 0; i < n; i++) {
    		if(arr[i] != dest[i]) count++;
    	}
    	return count;
    }
    
    
    

	
	public static void main(String[] args)  throws IOException{
    	BufferedWriter output = new BufferedWriter(new OutputStreamWriter(System.out));
		FastReader s = new FastReader();
		int t = s.nextInt();
		
		
    	for(int z = 0; z < t; z++) {
    		int n = s.nextInt();
    		String str1 = s.next();
    		String str2 = s.next();
    		int count1  = 0; int count2 = 0;
    		for(int i = 0; i < n; i++) {
    			if(str1.charAt(i) == '1') count1++;
    			if(str2.charAt(i) == '1') count2++;
    		}
    		if(!(count1 == count2 || n - count1 + 1 == count2)) {
    			System.out.println(-1);
    			continue;
    		}
    		
    		int[] arr = new int[n];
    		int[] dest = new int[n];
    		for(int i = 0; i < n; i++) {
    			arr[i] = str1.charAt(i) - '0';
    			dest[i] = str2.charAt(i) - '0';
    		}
    		int ans1 = count(arr, dest);
    		int ans2 = Integer.MAX_VALUE;
    		int flag = 0;
    		for(int i = 0; i < n; i++) {
    			if(flag == 0 && arr[i] == 1 && dest[i] == 1) {
    				flag = 1;
    			}
    			else {
    				arr[i] = 1 - arr[i];
    			}
    		}
    		//System.out.println(ans1);
    		if(flag == 1) {
    			ans2 = Math.min(ans2, count(arr, dest) + 1);
    		}
    		else {
    			for(int i = 0; i < n; i++) {
    				if(str1.charAt(i) == '1' && flag == 0) {
    					flag = 1; arr[i] = 1;
    					break;	
    				}
    			}
    			
    		}
    		if(flag == 1) ans2 = Math.min(ans2, count(arr, dest) + 1);
    		int ans = Integer.MAX_VALUE;
    		if(count1 == count2) ans = Math.min(ans, ans1);
    		if(n - count1 + 1 == count2) ans = Math.min(ans, ans2);
    		System.out.println(ans);
    		
    		
    		
    	}
    	//output.flush();

    }
}



