import java.util.*;
public class Main {

	public static void main(String[] args){
    	 Scanner scn = new Scanner(System.in);
    	 int t = scn.nextInt();
    	 StringBuilder sb = new StringBuilder("");
    	 for(int A=0 ; A<t ; A++){
    		 int n = scn.nextInt();
    		 String[] arr = new String[n];
    		 Map<String, Integer> hm = new HashMap<>();
    		 int min = 4;
    		 for(int i=0 ; i<n ; i++) {
    			 arr[i] = scn.next();
    			 hm.put(arr[i], arr[i].length());
    			 min = Math.min(min, arr[i].length());
    		 }
    		 if(min == 1)
    			 sb.append("YES" + "\n");
    		 else {
    			 boolean flag = false;
    			 for(int i=0 ; i<n ; i++) {
    				 if(arr[i].length() == 2 && arr[i].charAt(0) == arr[i].charAt(1)) {
    					 flag = true;
    					 break;
    				 }
    				 String s = "";
    				 for(int j=arr[i].length()-1 ; j>=0 ; j--)
    					 s += arr[i].charAt(j);
    				 if(hm.containsKey(s)) {
    					 flag = true;
    					 break;
    				 }
    				 if(arr[i].length() < 3) {
    					 for(int k=0 ; k<26 ; k++) {
    						 String str = (char)(k + 'a') + s;
    						 if(hm.containsKey(str)) {
    							 flag= true;
    							 break;
    						 }
    					 }
    				 }
    				 if(arr[i].length() == 3) {
    					 String str = s.substring(1, s.length());
    					 if(hm.containsKey(str)){
    						 flag = true;
    						 break;
    					 }
    				 }
    				 hm.remove(arr[i]);
    			 }
    			 if(flag)
    				 sb.append("YES" + "\n");
    			 else
    				 sb.append("NO" + "\n");
    		 }
    	 }
    	 System.out.println(sb);
	}
}
