import java.io.*;
import java.util.*;
		
	public class Aqueous {
		
		
		static MyScanner sc = new MyScanner();
			
		public static void main(String[] args) {
			
			int t = sc.nextInt();
			
				while(t-->0) {
					
					int n = sc.nextInt();
					
						String s[] = new String[n];
						
							for(int i = 0; i<n; i++) {
								s[i] = sc.next();
							}
							
							int ans = Integer.MIN_VALUE;
							
							for(char c = 'a'; c<='e'; c++) {
								
								int ls[] = new int[n];
								
									for(int i = 0; i<n; i++) {
										
										String temp = s[i];
										int delta = 0;
										
											for(int j = 0; j<temp.length(); j++) {
												if(temp.charAt(j)==c) {
													delta++;
												}
												else {
													delta--;
												}
											}
											
											ls[i] = delta;
									}
									
									Arrays.sort(ls);
									
									int cur = 0;
									int score= 0;
									
									for(int k = n-1; k>=0; k--) {
										if(cur+ls[k]>=1) {
											cur+=ls[k];
											score++;
										}
									}
									
									ans = Math.max(ans, score);
							}
							
							System.out.println(ans);
						
				}
		}
		
		
		
		
		

		
		
	    
	}
