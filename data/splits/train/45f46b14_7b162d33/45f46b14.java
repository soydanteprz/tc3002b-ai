import java.util.*;
public class Main {

	public static void main(String[] args){
    	 Scanner scn = new Scanner(System.in);
    	 int t = scn.nextInt();
    	 StringBuilder gs = new StringBuilder("");
    	 for(int Z=0 ; Z<t ; Z++){
    		 int n = scn.nextInt();
    		 String[] str = new String[n];
    		 Set<String> hs = new HashSet<>();
    		 int minLen = Integer.MAX_VALUE;
    		 for(int i=0 ; i<n ; i++) {
    			 str[i] = scn.next();
    			 hs.add(str[i]);
    			 minLen = Math.min(minLen, str[i].length());
    		 }
    		 if(minLen == 1)
    			 gs.append("YES" + "\n");
    		 else {
    			 boolean mark = false;
    			 for(int i=0 ; i<n ; i++) {
    				 if(str[i].length() == 2 && str[i].charAt(0) == str[i].charAt(1)) {
    					 mark = true;
    					 break;
    				 }
    				 String rev = "";
    				 for(int j=str[i].length()-1 ; j>=0 ; j--)
    					 rev += str[i].charAt(j);
    				 if(hs.contains(rev)) {
    					 mark = true;
    					 break;
    				 }
    				 if(str[i].length() == 3) {
    					 String ans = rev.substring(1, rev.length());
    					 if(hs.contains(ans)){
    						 mark = true;
    						 break;
    					 }
    				 }
    				 if(str[i].length() < 3) {
    					 for(int k=0 ; k<26 ; k++) {
    						 String str1 = (char)(k + 'a') + rev;
    						 if(hs.contains(str1)) {
    							 mark= true;
    							 break;
    						 }
    					 }
    				 }
    				
    				 hs.remove(str[i]);
    			 }
    			 if(!mark)
    				 gs.append("NO" + "\n");
    			 else
    				 gs.append("YES" + "\n");
    		 }
    	 }
    	 System.out.println(gs);
	}
}
